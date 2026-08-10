#!/usr/bin/env python
"""Submit a Deci/NEL eval with four TP2 vLLM instances per 8-GPU node.

This script is intentionally a small shim around deci-evals. It keeps the
normal config construction, secrets handling, ExecDB writes, and Slurm submit
path, but patches NEL's generated Slurm deployment section so TP2 instances are
packed onto each allocated 8-GPU node:

  node 0: CUDA_VISIBLE_DEVICES=0,1 port 8000
          CUDA_VISIBLE_DEVICES=2,3 port 8001
          CUDA_VISIBLE_DEVICES=4,5 port 8002
          CUDA_VISIBLE_DEVICES=6,7 port 8003

For multi-node runs, the same four local ports are used on each node and the
proxy backends point at each node IP plus those ports.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import os
import pathlib
import re
import shlex

from jinja2 import Environment, FileSystemLoader
from omegaconf import OmegaConf, open_dict


PORT_ENV = "PACKED_TP2_PORT"


PACKED_PROXY_SCRIPT = r"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import itertools
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backends: list[tuple[str, int]] = []
    backend_cycle = None
    backend_lock = threading.Lock()

    def log_message(self, fmt: str, *args) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), fmt % args), flush=True)

    def _next_backend(self) -> tuple[str, int]:
        with self.backend_lock:
            return next(self.backend_cycle)

    def _forward(self) -> None:
        if self.path == "/health":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }

        errors: list[str] = []
        for _ in range(len(self.backends)):
            host, port = self._next_backend()
            try:
                conn = http.client.HTTPConnection(host, port, timeout=600)
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                payload = resp.read()
                self.send_response(resp.status, resp.reason)
                for key, value in resp.getheaders():
                    if key.lower() in HOP_BY_HOP_HEADERS:
                        continue
                    if key.lower() == "content-length":
                        continue
                    self.send_header(key, value)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                conn.close()
                return
            except Exception as exc:
                errors.append(f"{host}:{port}: {exc}")

        payload = ("all backends failed: " + "; ".join(errors) + "\n").encode()
        self.send_response(502)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _forward
    do_POST = _forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backends", required=True)
    args = parser.parse_args()

    backends = []
    for item in args.backends.split(","):
        if not item:
            continue
        host, port = item.rsplit(":", 1)
        backends.append((host, int(port)))
    if not backends:
        raise SystemExit("no proxy backends configured")

    ProxyHandler.backends = backends
    ProxyHandler.backend_cycle = itertools.cycle(backends)
    print(f"packed python proxy listening on {args.listen}:{args.port}", flush=True)
    print("backends: " + ",".join(f"{host}:{port}" for host, port in backends), flush=True)
    ThreadingHTTPServer((args.listen, args.port), ProxyHandler).serve_forever()


if __name__ == "__main__":
    main()
"""


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _packed_enabled() -> bool:
    return os.environ.get("PACKED_TP2", "1") == "1"


def _instances_per_node() -> int:
    return _env_int("PACKED_TP2_INSTANCES_PER_NODE", 4)


def _actual_num_nodes() -> int:
    return _env_int("PACKED_TP2_NUM_NODES", 1)


def _port_base() -> int:
    return _env_int("PACKED_TP2_PORT_BASE", 8000)


def _cuda_groups() -> list[str]:
    raw = os.environ.get("PACKED_TP2_CUDA_GROUPS", "0,1;2,3;4,5;6,7")
    groups = [group.strip() for group in raw.split(";") if group.strip()]
    expected = _instances_per_node()
    if len(groups) != expected:
        raise ValueError(
            "PACKED_TP2_CUDA_GROUPS must contain exactly "
            f"{expected} semicolon-separated groups, got {groups!r}"
        )
    for group in groups:
        devices = [device.strip() for device in group.split(",") if device.strip()]
        if len(devices) != 2:
            raise ValueError(f"Packed TP2 CUDA group {group!r} does not contain 2 GPUs")
    return groups


def _require_tp2(command: str) -> None:
    if not re.search(r"--tensor-parallel-size(?:=|\s+)2(?:\s|$)", command):
        raise ValueError(
            "Packed launcher requires deployment.command to contain "
            "--tensor-parallel-size=2"
        )


def _command_with_dynamic_port(command: str) -> str:
    _require_tp2(command)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}${{{PORT_ENV}}}"

    updated, count = re.subn(r"(--port(?:=|\s+))\d+", repl, command, count=1)
    if count != 1:
        raise ValueError("deployment.command must contain a numeric --port")
    return updated


def _copy_config(cfg):
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(copied, False)
    return copied


def _install_nel_patches() -> None:
    if not _packed_enabled():
        return

    import nemo_evaluator_launcher.executors.slurm.executor as slurm_executor
    from nemo_evaluator_launcher.common.helpers import _str_to_echo_command

    actual_nodes = _actual_num_nodes()
    instances_per_node = _instances_per_node()
    total_instances = actual_nodes * instances_per_node
    port_base = _port_base()
    cuda_groups = _cuda_groups()

    orig_create = slurm_executor._create_slurm_sbatch_script
    orig_wait = slurm_executor._get_wait_for_server_handler

    def packed_haproxy_config_with_placeholders(cfg):
        template_dir = pathlib.Path(slurm_executor.__file__).parent
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("proxy.cfg.template")

        nodes = []
        for node_idx in range(actual_nodes):
            for local_idx in range(instances_per_node):
                nodes.append(
                    {
                        "ip": f"{{IP_{node_idx}}}",
                        "port": port_base + local_idx,
                    }
                )

        proxy_config = cfg.execution.get("proxy", {}).get("config", {})
        health_check_path = proxy_config.get(
            "health_check_path", cfg.deployment.endpoints.get("health", "/health")
        )
        health_check_status = proxy_config.get("health_check_status", 200)
        haproxy_port = proxy_config.get("haproxy_port", 5009)

        return template.render(
            haproxy_port=haproxy_port,
            health_check_path=health_check_path,
            health_check_status=health_check_status,
            nodes=nodes,
        )

    def packed_wait_for_server_handler(
        ip_list: str,
        port: int,
        health_check_path: str,
        service_name: str = "server",
        check_pid: bool = False,
    ):
        if service_name != "server":
            return orig_wait(ip_list, port, health_check_path, service_name, check_pid)

        pid_check = ""
        if check_pid:
            pid_check = (
                'for _check_pid in "${SERVER_PIDS[@]}"; do '
                'kill -0 "$_check_pid" 2>/dev/null || '
                '{ echo "Server process $_check_pid died"; exit 1; }; done'
            )

        return f"""date
# wait for the {service_name} to initialize
for i in "${{!HEAD_NODE_IPS[@]}}"; do
  ip="${{HEAD_NODE_IPS[$i]}}"
  server_port="${{HEAD_NODE_PORTS[$i]}}"
  echo "Waiting for {service_name} on $ip:$server_port..."
  while [[ "$(curl -s -o /dev/null -w "%{{http_code}}" http://$ip:$server_port{health_check_path})" != "200" ]]; do
    {pid_check}
    sleep 5
  done
  echo "{service_name} ready on $ip:$server_port!"
done
date""".strip()

    def packed_deployment_srun_command(
        cfg,
        deployment_mounts_list,
        remote_task_subdir,
        deployment_env_var_names: list[str] | None = None,
    ):
        if int(cfg.execution.num_instances) != total_instances:
            raise ValueError(
                "Packed TP2 expected execution.num_instances="
                f"{total_instances}, got {cfg.execution.num_instances}"
            )

        command = _command_with_dynamic_port(str(cfg.deployment.command))
        pre_cmd: str = cfg.deployment.get("pre_cmd") or ""
        is_potentially_unsafe = bool(pre_cmd)
        debug_comment = ""

        if pre_cmd:
            create_pre_script_cmd = _str_to_echo_command(
                pre_cmd, filename="deployment_pre_cmd.sh"
            )
            debug_comment += create_pre_script_cmd.debug + "\n\n"

        create_script_cmd = _str_to_echo_command(
            command, filename="deployment_cmd.sh"
        )
        debug_comment += create_script_cmd.debug + "\n\n"

        env_setup = "export PROC_ID=${SLURM_PROCID:-0} NODES_PER_INSTANCE=1"
        inner_env_setup = (
            f"{env_setup} && "
            'export CUDA_DEVICE_ORDER=PCI_BUS_ID '
            'CUDA_VISIBLE_DEVICES="${PACKED_TP2_CUDA_GROUP}" '
            'NVIDIA_VISIBLE_DEVICES="${PACKED_TP2_CUDA_GROUP}" && '
            'echo "Container packed TP2 env: '
            'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} '
            'NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} '
            f'{PORT_ENV}=${{{PORT_ENV}}}"'
        )
        script = (
            f"{inner_env_setup} && "
            f"{create_script_cmd.cmd} && bash deployment_cmd.sh"
        )
        if pre_cmd:
            create_pre_script_cmd = _str_to_echo_command(
                pre_cmd, filename="deployment_pre_cmd.sh"
            )
            script = (
                f"{inner_env_setup} && "
                f"{create_pre_script_cmd.cmd} && "
                "source deployment_pre_cmd.sh && "
                f"{create_script_cmd.cmd} && bash deployment_cmd.sh"
            )

        env_names = list(deployment_env_var_names or [])
        for name in (
            "MASTER_IP",
            "ALL_NODE_IPS",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "PACKED_TP2_CUDA_GROUP",
            PORT_ENV,
        ):
            if name not in env_names:
                env_names.append(name)

        quoted_script = shlex.quote(script)
        group_array = " ".join(shlex.quote(group) for group in cuda_groups)

        s = "# deployment server\n"
        s += "# Get node IPs\n"
        s += 'NODELIST="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"\n'
        s += 'if command -v scontrol >/dev/null 2>&1 && [[ -n "${NODELIST}" ]]; then\n'
        s += '  nodes=( $(scontrol show hostnames "${NODELIST}") )\n'
        s += "else\n"
        s += '  nodes=( "$(hostname)" )\n'
        s += "fi\n"
        s += 'nodes_array=("${nodes[@]}")  # Ensure nodes are stored properly\n'
        s += 'if [[ ${#nodes_array[@]} -eq 0 ]]; then nodes_array=( "$(hostname)" ); fi\n'
        s += f"if [[ ${{#nodes_array[@]}} -lt {actual_nodes} ]]; then\n"
        s += (
            f'  echo "Packed TP2 expected {actual_nodes} allocated nodes, '
            'got ${#nodes_array[@]}" >&2\n'
        )
        s += "  exit 1\n"
        s += "fi\n"
        s += 'export NODES_IPS_ARRAY=($(for node in "${nodes_array[@]}"; do srun --nodelist="$node" --ntasks=1 --nodes=1 hostname --ip-address; done))\n'
        s += 'echo "Node IPs: ${NODES_IPS_ARRAY[@]}"\n'
        s += 'export ALL_NODE_IPS=$(IFS=,; echo "${NODES_IPS_ARRAY[*]}")\n'
        s += f"PACKED_TP2_CUDA_GROUPS=({group_array})\n"
        s += "HEAD_NODE_IPS=()\n"
        s += "HEAD_NODE_PORTS=()\n"
        s += "SERVER_PIDS=()\n"

        if debug_comment:
            s += "# Debug contents of packed deployment command\n"
            s += "\n".join("# " + line for line in debug_comment.splitlines())
            s += "\n\n"

        s += f"for ((node_i=0; node_i<{actual_nodes}; node_i++)); do\n"
        s += f"  for ((local_i=0; local_i<{instances_per_node}; local_i++)); do\n"
        s += f"    g=$((node_i * {instances_per_node} + local_i))\n"
        s += '    INSTANCE_NODE="${nodes_array[$node_i]}"\n'
        s += '    MASTER_IP="${NODES_IPS_ARRAY[$node_i]}"\n'
        s += f"    {PORT_ENV}=$(({port_base} + local_i))\n"
        s += '    PACKED_TP2_CUDA_GROUP="${PACKED_TP2_CUDA_GROUPS[$local_i]}"\n'
        s += '    CUDA_VISIBLE_DEVICES="${PACKED_TP2_CUDA_GROUPS[$local_i]}"\n'
        s += '    NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"\n'
        s += '    HEAD_NODE_IPS+=("$MASTER_IP")\n'
        s += f'    HEAD_NODE_PORTS+=("${{{PORT_ENV}}}")\n'
        s += (
            "    export MASTER_IP PACKED_TP2_CUDA_GROUP "
            "CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES "
        )
        s += f"{PORT_ENV}\n"
        s += (
            '    echo "Packed TP2 instance $g: node=$INSTANCE_NODE '
            'MASTER_IP=$MASTER_IP CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES '
            f'port=${{{PORT_ENV}}}"\n'
        )
        s += "    srun --mpi pmix --overlap "
        s += '--nodelist "$INSTANCE_NODE" --nodes 1 --ntasks 1 '
        s += f"--container-image {cfg.deployment.image} "
        if deployment_mounts_list:
            s += "--container-mounts {} ".format(",".join(deployment_mounts_list))
        if not cfg.execution.get("mounts", {}).get("mount_home", True):
            s += "--no-container-mount-home "
        s += "--output {} ".format(remote_task_subdir / "logs" / "server-${g}-%A-%t.log")
        s += f"--container-env {','.join(sorted(env_names))} "
        s += f"bash -c {quoted_script} &\n"
        s += "    SERVER_PIDS+=($!)\n"
        s += "  done\n"
        s += "done\n\n"
        s += 'echo "HEAD_NODE_IPS: ${HEAD_NODE_IPS[@]}"\n'
        s += 'echo "HEAD_NODE_PORTS: ${HEAD_NODE_PORTS[@]}"\n'
        s += "SERVER_PID=${SERVER_PIDS[0]}  # reference to first instance PID for health check\n\n"

        return s, is_potentially_unsafe, debug_comment

    def packed_create_slurm_sbatch_script(
        cfg,
        task,
        eval_image,
        remote_task_subdir,
        invocation_id,
        job_id,
    ):
        if int(cfg.execution.num_instances) != total_instances:
            raise ValueError(
                "Packed TP2 expected execution.num_instances="
                f"{total_instances}, got {cfg.execution.num_instances}"
            )

        fake_cfg = _copy_config(cfg)
        with open_dict(fake_cfg.execution):
            fake_cfg.execution.num_nodes = total_instances
            fake_cfg.execution.ntasks_per_node = 1
            if "deployment" not in fake_cfg.execution:
                fake_cfg.execution.deployment = {}
            fake_cfg.execution.deployment.n_tasks = total_instances

        result = orig_create(
            fake_cfg,
            task,
            eval_image,
            remote_task_subdir,
            invocation_id,
            job_id,
        )
        cmd = result.cmd.replace(
            f"#SBATCH --nodes {total_instances}\n",
            f"#SBATCH --nodes {actual_nodes}\n",
            1,
        )

        _verify_script(cmd, str(cfg.deployment.command))
        print(
            "PACKED_TP2 verified generated run.sub: "
            f"nodes={actual_nodes}, instances_per_node={instances_per_node}, "
            f"total_instances={total_instances}, cuda_groups={cuda_groups}, "
            f"ports={list(range(port_base, port_base + instances_per_node))}, tp=2",
            flush=True,
        )
        return dataclasses.replace(result, cmd=cmd)

    def _verify_script(script: str, command: str) -> None:
        _require_tp2(command)
        proxy_cfg = packed_haproxy_config_with_placeholders(OmegaConf.create({
            "execution": {
                "proxy": {
                    "config": {
                        "haproxy_port": 5009,
                        "health_check_path": "/health",
                        "health_check_status": 200,
                    }
                }
            },
            "deployment": {"endpoints": {"health": "/health"}},
        }))
        for node_idx in range(actual_nodes):
            for local_idx in range(instances_per_node):
                backend = f"{{IP_{node_idx}}}:{port_base + local_idx}"
                if backend not in proxy_cfg:
                    raise AssertionError(f"Missing packed proxy backend {backend}")
        for group in cuda_groups:
            if group not in script:
                raise AssertionError(f"Missing CUDA group {group} in run.sub")
        if f"{PORT_ENV}=$(({port_base} + local_i))" not in script:
            raise AssertionError("Missing dynamic packed TP2 port assignment")
        if "CUDA_VISIBLE_DEVICES" not in script:
            raise AssertionError("Missing CUDA_VISIBLE_DEVICES in run.sub")

    slurm_executor._generate_haproxy_config_with_placeholders = (
        packed_haproxy_config_with_placeholders
    )
    slurm_executor._get_wait_for_server_handler = packed_wait_for_server_handler
    slurm_executor._generate_deployment_srun_command = packed_deployment_srun_command
    slurm_executor._create_slurm_sbatch_script = packed_create_slurm_sbatch_script

    def packed_python_proxy_srun_command(cfg, remote_task_subdir):
        proxy_config = cfg.execution.get("proxy", {}).get("config", {})
        haproxy_port = proxy_config.get("haproxy_port", 5009)
        proxy_image = cfg.execution.get("proxy", {}).get("image", cfg.deployment.image)
        encoded_proxy = base64.b64encode(PACKED_PROXY_SCRIPT.encode()).decode()

        s = "# packed Python proxy server\n"
        s += (
            'PACKED_PROXY_BACKENDS=$(for i in "${!HEAD_NODE_IPS[@]}"; do '
            'printf \'%s:%s\\n\' "${HEAD_NODE_IPS[$i]}" "${HEAD_NODE_PORTS[$i]}"; '
            "done | paste -sd, -)\n"
        )
        s += 'echo "Packed proxy backends: ${PACKED_PROXY_BACKENDS}"\n'
        s += "export PACKED_PROXY_BACKENDS\n"
        s += "srun --mpi pmix --overlap "
        s += '--nodelist "${PRIMARY_NODE}" --nodes 1 --ntasks 1 '
        s += f"--container-image {proxy_image} "
        if not cfg.execution.get("mounts", {}).get("mount_home", True):
            s += "--no-container-mount-home "
        s += f"--container-mounts /lustre:/lustre,/scratch:/scratch,/tmp:/tmp "
        s += f"--output {remote_task_subdir}/logs/proxy-%A.log "
        s += "--container-env PACKED_PROXY_BACKENDS "
        s += "bash -c "
        proxy_cmd = (
            f"echo {shlex.quote(encoded_proxy)} | base64 -d > packed_python_proxy.py "
            f"&& python3 -u packed_python_proxy.py --port {haproxy_port} "
            '--backends "$PACKED_PROXY_BACKENDS"'
        )
        s += f"{shlex.quote(proxy_cmd)} &\n"
        s += "PROXY_PID=$!  # capture the PID of the proxy background srun process\n"
        s += packed_wait_for_server_handler(
            "127.0.0.1", haproxy_port, "/health", "Proxy", check_pid=False
        )
        s += "\n\n"
        return s

    slurm_executor._get_proxy_server_srun_command = packed_python_proxy_srun_command


def _patch_deci_adapter(dry_run: bool) -> None:
    from deci_evals.evaluate_utils import NemoEvaluatorAdapter

    orig_lazy = NemoEvaluatorAdapter._lazy_import_nel

    def lazy_with_packed_patches(self, force_reimport: bool = False):
        run_config, run_eval = orig_lazy(self, force_reimport=force_reimport)
        _install_nel_patches()
        if not dry_run:
            return run_config, run_eval

        def run_eval_dry(cfg, *args, **kwargs):
            kwargs["dry_run"] = True
            return run_eval(cfg, *args, **kwargs)

        return run_config, run_eval_dry

    NemoEvaluatorAdapter._lazy_import_nel = lazy_with_packed_patches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a deci-evals run with packed TP2 Slurm layout."
    )
    parser.add_argument("checkpoint_path")
    parser.add_argument("evaluator_name")
    parser.add_argument("output_path")
    parser.add_argument("--tasks", action="append", default=[])
    parser.add_argument("--cluster")
    parser.add_argument("--account")
    parser.add_argument("--dirty-tag")
    parser.add_argument("--skip-ssh", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--exact-output-path", action="store_true")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument(
        "--packed-script-dry-run",
        action="store_true",
        help="Generate and print NEL Slurm scripts without submitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dry_run = args.packed_script_dry_run or os.environ.get("PACKED_TP2_DRY_RUN") == "1"

    _patch_deci_adapter(dry_run=dry_run)

    from deci_evals.cli.evaluate import evaluate

    evaluate(
        checkpoint_path=pathlib.Path(args.checkpoint_path),
        evaluator_name=args.evaluator_name,
        output_path=pathlib.Path(args.output_path),
        exact_output_path=args.exact_output_path,
        config_only=args.config_only,
        tasks=args.tasks or None,
        overrides=args.overrides,
        cluster=args.cluster,
        account=args.account,
        dirty_tag=args.dirty_tag,
        skip_ssh=args.skip_ssh,
    )


if __name__ == "__main__":
    main()
