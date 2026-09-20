"""Synthetic metric adapters for contract tests; no generator or fitting."""

from copy import deepcopy

from genode.gico.clocks import materialize


def evaluator_for(rows, contexts, *, factor=0.9, replicates=4):
    def evaluate(candidate):
        measured = []
        for anchor in rows:
            if anchor["split"] != "validation" or anchor["schedule_key"] != "uniform":
                continue
            count = replicates if candidate.student_kind == "GICO-sto-policy" else 1
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
                if candidate.student_kind == "GICO-sto-policy" and anchor["ensemble_size"] > 1:
                    student.pop("density_mass")
                    student.pop("time_grid")
                    student["sample_clocks"] = clocks
                else:
                    student.update(clocks[0])
                measured.extend((uniform, student))
        return measured

    return evaluate


def fixture_history(students, evidence, rows, contexts, *, teacher, conditioning=None, step=2000, coefficient=0.01):
    """Synthetic complete checkpoint records for artifact-only tests."""
    from genode.gico.deterministic_selection import DETERMINISTIC_SELECTION_PROTOCOL
    from genode.gico.evidence import content_hash
    from genode.gico.selection import candidate_fingerprint, teacher_fingerprint
    from genode.gico.stochastic_selection import STOCHASTIC_SELECTION_PROTOCOL
    from genode.gico.training import score_coefficient

    selected, histories = {}, {}
    for kind, model in students.items():
        cadence = 10 if kind == "GICO-det-policy" else 100
        history = []
        for checkpoint in range(cadence, step + 1, cadence):
            record = {
                "step": checkpoint,
                "coefficient": score_coefficient(checkpoint - 1, step, coefficient),
                "validation_distillation": 1.0,
            }
            if record["coefficient"] > 0:
                record.update(
                    selection_protocol=DETERMINISTIC_SELECTION_PROTOCOL
                    if kind == "GICO-det-policy"
                    else STOCHASTIC_SELECTION_PROTOCOL,
                    predicted_utility=checkpoint / step,
                    selection_checkpoint_id=candidate_fingerprint(
                        model, conditioning or evidence.conditioning, kind, checkpoint
                    ),
                    selection_teacher_fingerprint=teacher_fingerprint(teacher, evidence.conditioning, 20, 0.05),
                    selection_contexts=sorted({r["context_id"] for r in rows if r["split"] == "validation"}),
                    selection_groups=len(evidence.groups("validation")),
                    predictions_sha256=content_hash([kind, checkpoint]),
                    clock_replicates=4,
                    kl_samples_per_reference=32,
                )
            history.append(record)
        histories[kind], selected[kind] = history, history[-1]
    return {"students": histories, "student_selection": selected}
