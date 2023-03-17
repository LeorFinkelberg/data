#! python

import click
import re
import pandas as pd
import typing as t
import time
import pyscipopt
from functools import wraps
import pathlib2
import itertools

from tqdm import tqdm

CONTINUOUS = "CONTINUOUS"
BINARY = "BINARY"
INTEGER = "INTEGER"
UP_BOUND_TYPE = "UP"
LO_BOUND_TYPE = "LO"
FX_BOUND_TYPE = "FX"
BV_BOUND_TYPE = "BV"
STATUS_OPTIMAL_SOL = "optimal"
HIGHS = "HIGHS"
SKIP_LINES_HIGHS_SOL_FILE = 7
STOP_LINE_HIGHS_SOL_FILE = "# Row"
SCIP = "SCIP"


def timer(f: t.Callable):
    """
    Measures the execution time of the function
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start_record = time.process_time()
        f(*args, **kwargs)
        print(f"Full calculation time: {(time.process_time() - start_record) / 60:.5f} [min]")

    return wrapper


def parse_sol_file(path_to_sol_file: pathlib2.Path) -> t.Dict[str, float]:
    """
    Parses sol-files
    """

    def _detect_sol_file_format(path_to_sol_file: pathlib2.Path) -> str:
        """
        Detects sol-files format
        """
        # TODO: implement logic for sol-file format detector
        with open(str(path_to_sol_file), encoding="utf-8") as sol_file:
            for line in sol_file:
                if line.startswith("Model status"):
                    return HIGHS
                elif line.startswith("objective value"):
                    return SCIP

    sol: t.Dict[str, float] = {}

    with open(str(path_to_sol_file), encoding="utf-8") as sol_file:
        sol_file_format: str = _detect_sol_file_format(path_to_sol_file)

        if sol_file_format == HIGHS:
            for line in itertools.islice(sol_file, SKIP_LINES_HIGHS_SOL_FILE, None):
                if line.startswith(STOP_LINE_HIGHS_SOL_FILE):
                    break
                var_name, value = line.split()
                sol[var_name] = float(value)

        elif sol_file_format == SCIP:
            for line in itertools.dropwhile(
                    lambda line: line.startswith(("sol", "obj", "#")),
                    sol_file,
            ):
                var_name, value = line.split()[:2]
                sol[var_name] = float(value)
        else:
            raise

    return sol


@click.command()
@click.option(
    "--path-to-problem",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=pathlib2.Path,
    ),
    help="path to lp/mps-file",
)
@click.option(
    "--path-to-relax-settings",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=pathlib2.Path,
    ),
    help="path to SCIP RELAX set-file",
)
@click.option(
    "--path-to-milp-settings",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=pathlib2.Path,
    ),
    help="path to SCIP MILP set-file",
)
@click.option(
    "--path-to-presolving-settings",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=pathlib2.Path,
    ),
    help="path to SCIP set-file",
)
@click.option(
    "--path-to-sols-dir",
    required=True,
    type=click.Path(
        exists=True,
        path_type=pathlib2.Path,
    ),
    help="path to sol-files dir",
)
@timer
def main(
    path_to_problem: pathlib2.Path,
    path_to_relax_settings: pathlib2.Path,
    path_to_presolving_settings: pathlib2.Path,
    path_to_milp_settings: pathlib2.Path,
    path_to_sols_dir: pathlib2.Path,
):
    # RELAX
    print("RELAX")
    model = pyscipopt.Model()
    model.readProblem(path_to_problem)
    # model.readParams("./data/set_files/scip_7.0.3_emulator.set")
    model.readParams(path_to_relax_settings)

    _all_vars: t.List[pyscipopt.scip.Variable] = model.getVars()
    for var in tqdm(_all_vars):
        model.chgVarType(var, CONTINUOUS)

    model.optimize()
    status = model.getStatus()

    problem_name: str = path_to_problem.name
    suffix: str = path_to_problem.suffix
    path_to_relax_sol = path_to_sols_dir.joinpath(f"{problem_name.replace(suffix, '')}_relax.sol")

    if status == STATUS_OPTIMAL_SOL:
        best_relax_sol = model.getBestSol()
        model.writeSol(best_relax_sol, path_to_relax_sol, write_zeros=True)
    else:
        raise

    # PRESOLVING
    print("PRESOLVING")
    model = pyscipopt.Model()
    model.readProblem(path_to_problem)
    # model.readParams("./data/set_files/scip_7.0.3_emulator.set")
    model.readParams(path_to_presolving_settings)
    model.presolve()

    path_to_problem_parent: pathlib2.Path = path_to_problem.parent
    problem_name_after_presolving = f"{problem_name.replace(suffix, '')}_after_presolving.trans{suffix}"
    path_to_problem_after_presolving = path_to_problem_parent.joinpath(problem_name_after_presolving)
    model.writeProblem(path_to_problem_after_presolving, trans=True)

    # FIX and MILP
    print("FIX and MILP")
    PATTERN = re.compile(r"^t_")
    model = pyscipopt.Model()
    model.readProblem(path_to_problem)
    # model.readParams("./data/set_files/scip_7.0.3_emulator.set")
    model.readParams(path_to_milp_settings)

    _all_vars: t.List[pyscipopt.scip.Variable] = model.getVars()
    _all_var_names: t.List[str] = [var.name for var in _all_vars]
    all_vars = pd.Series(_all_vars, index=_all_var_names)
    _bin_int_vars: t.List[pyscipopt.scip.Variable] = [var for var in _all_vars if var.vtype() != CONTINUOUS]
    _bin_int_var_names: t.List[str] = [var.name for var in _bin_int_vars]

    var_name_to_obj_var: t.Dict[str, pyscipopt.scip.Variable] = {
        var.name: var for var in _all_vars
    }

    (
        bound_type_lo_cnt,
        bound_type_up_cnt,
        bound_type_fx_cnt,
        bound_type_bv_cnt,
    ) = (0,) * 4

    relax_sol = pd.Series(parse_sol_file(path_to_relax_sol), name="RELAX").loc[_bin_int_var_names]
    print(relax_sol.value_counts())
    mask = relax_sol == 0.0
    var_names_for_fix = relax_sol.loc[mask].index.to_list()

    bounded_var_names= set()
    with open(path_to_problem_after_presolving) as problem_after_presolving:
        for line in problem_after_presolving:
            line = line.strip()
            if line.startswith(("UP Bound", "LO Bound",)):
                bound_type, _, var_name, _value = line.split()
                var_name = PATTERN.sub("", var_name)

                if var_name in var_name_to_obj_var.keys():
                    value = float(_value)
                    var = var_name_to_obj_var.get(var_name)

                    if bound_type == LO_BOUND_TYPE:
                        bounded_var_names.add(var_name)
                        model.chgVarLb(var, value)
                        bound_type_lo_cnt += 1

                    elif bound_type == UP_BOUND_TYPE:
                        # print(bound_type, var_name)
                        bounded_var_names.add(var_name)
                        model.chgVarUb(var, value)
                        bound_type_up_cnt += 1
                    # elif bound_type == FX_BOUND_TYPE and var.vtype() == CONTINUOUS and value < 100_000:
                    elif bound_type == FX_BOUND_TYPE and value < 100_000:
                        # print(f"<FX>: {var.name} ({var.vtype()}) = {value}")
                        model.fixVar(var, value)
                        bound_type_fx_cnt += 1
                    else:
                        continue

            """
            elif line.startswith(("BV Bound",)):
                bound_type, _, var_name = line.split()
                var_name = PATTERN.sub("", var_name)

                if var_name in var_name_to_obj_var.keys(): 
                    # print(bound_type, var_name)
                    bounded_var_names.add(var_name)
                    var = var_name_to_obj_var.get(var_name)
                    model.chgVarLb(var, 0.0)
                    model.chgVarUb(var, 1.0)
                    bound_type_bv_cnt += 1
            """
    
    # bounded_vars = [var_name_to_obj_var.get(var_name) for var_name in bounded_var_names]
    # bins_ints_bounded_vars = set([(var.name, var.vtype()) for var in bounded_vars])
    # var_names_for_fix = list(set(var_names_for_fix).difference(bins_ints_bounded_vars))

    for var in tqdm(all_vars.loc[var_names_for_fix]):
        model.fixVar(var, relax_sol.loc[var.name])

    print(f"BOUND TYPE <LO>: {bound_type_lo_cnt}")
    print(f"BOUND TYPE <UP>: {bound_type_up_cnt}")
    print(f"BOUND TYPE <FX>: {bound_type_fx_cnt}")
    print(f"BOUND TYPE <BV>: {bound_type_bv_cnt}")

    model.presolve()
    print("STATISTICS")
    # model.printStatistics()


    model.optimize()
    best_milp_sol = model.getBestSol()
    path_to_milp_sol = path_to_sols_dir.joinpath(f"{problem_name.replace(suffix, '')}_milp.sol")
    model.writeSol(best_milp_sol, path_to_milp_sol, write_zeros=True)


if __name__ == "__main__":
    main()
