import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


def run_hydra_cli():
    import hydra
    from utils.utils import init_client, print_hyperlink

    @hydra.main(version_base=None, config_path="cfg", config_name="config")
    def main(cfg):
        workspace_dir = Path.cwd()
        logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
        logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
        logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
        logging.info(f"Using Algorithm: {cfg.algorithm}")

        client = init_client(cfg)
        long_ref_llm = hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
        short_ref_llm = hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
        crossover_llm = hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
        mutation_llm = hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None

        if cfg.algorithm == "reevo":
            from reevo import ReEvo as LHH
        elif cfg.algorithm == "ael":
            from baselines.ael.ga import AEL as LHH
        elif cfg.algorithm == "eoh":
            from baselines.eoh import EoH as LHH
        else:
            raise NotImplementedError

        if cfg.algorithm != "reevo":
            lhh = LHH(cfg, ROOT_DIR, client)
        else:
            lhh = LHH(
                cfg,
                ROOT_DIR,
                client,
                long_reflector_llm=long_ref_llm,
                short_reflector_llm=short_ref_llm,
                crossover_llm=crossover_llm,
                mutation_llm=mutation_llm,
            )

        best_code_overall, best_code_path_overall = lhh.evolve()
        logging.info(f"Best Code Overall: {best_code_overall}")
        best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
        logging.info(f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}")

        with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", "w", encoding="utf-8") as file:
            file.writelines(best_code_overall + "\n")
        test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
        test_script_stdout = "best_code_overall_val_stdout.txt"
        logging.info(f"Running validation script...: {print_hyperlink(test_script)}")
        with open(test_script_stdout, "w", encoding="utf-8") as stdout:
            subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
        logging.info(f"Validation script finished. Results are saved in {print_hyperlink(test_script_stdout)}.")

        with open(test_script_stdout, "r", encoding="utf-8") as file:
            for line in file.readlines():
                logging.info(line.strip())

    main()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from app import run_server

        run_server(port=int(os.environ.get("PORT", 3000)))
    else:
        run_hydra_cli()
