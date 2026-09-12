"""Synthetic metric adapters for contract tests; no generator or fitting."""

from copy import deepcopy

from genode.gico.clocks import materialize
from genode.gico.selection import evaluate_candidate


def cli_factory(config):
    return lambda candidate: []


def evaluator_for(rows, contexts, *, factor=0.9, replicates=4):
    def evaluate(candidate):
        measured = []
        for anchor in rows:
            if anchor["split"] != "validation" or anchor["schedule_key"] != "uniform":
                continue
            count = replicates if candidate.student_kind == "stochastic" else 1
            for rep in range(count):
                uniform = deepcopy(anchor)
                uniform["clock_replicate"] = rep
                student = deepcopy(uniform)
                student.update(schedule_key="student", selection_checkpoint_id=candidate.checkpoint_id)
                student["metrics"] = {k: v * factor for k, v in anchor["metrics"].items()}
                clocks = []
                for member in range(anchor["ensemble_size"]):
                    request = (
                        f"{anchor['context_id']}:{anchor['solver']}:{anchor['nfe']}:{anchor['seed']}:{rep}:{member}"
                    )
                    mass = candidate.density(
                        contexts[anchor["context_id"]], anchor["solver"], anchor["nfe"], seed=73, request_id=request
                    )
                    clocks.append(
                        {
                            "density_mass": mass.tolist(),
                            "time_grid": list(materialize(mass, anchor["solver"], anchor["nfe"])),
                            "clock_seed": 73,
                            "clock_request_id": request,
                        }
                    )
                if candidate.student_kind == "stochastic" and anchor["ensemble_size"] > 1:
                    student.pop("density_mass")
                    student.pop("time_grid")
                    student["sample_clocks"] = clocks
                else:
                    student.update(clocks[0])
                measured.extend((uniform, student))
        return measured

    return evaluate


def fixture_history(students, evidence, rows, contexts, *, conditioning=None, step=500, coefficient=0.01):
    selected = {}
    for kind, model in students.items():
        selected[kind] = {
            "step": step,
            "coefficient": coefficient,
            **evaluate_candidate(
                model,
                conditioning or evidence.conditioning,
                kind,
                step,
                coefficient,
                evaluator_for(rows, contexts),
                evidence,
                4,
            ),
        }
    return {"students": {k: [r] for k, r in selected.items()}, "student_selection": selected}
