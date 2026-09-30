"""Stage 0 contract tests for poc.types (plan.md sections 3 and 8)."""

import dataclasses

import pytest

import poc
from poc.types import (
    ActionSpec,
    Entity,
    EntityRole,
    Intervention,
    InterventionKind,
    RepairResult,
)


def _relocate(pid: str, entity: str, length: int = 1) -> Intervention:
    return Intervention(pid, InterventionKind.RELOCATE, entity, length=length, params=(0.0, 1.0))


def _result(interventions: tuple[Intervention, ...], F: int = 0, **overrides) -> RepairResult:
    fields = dict(
        interventions=interventions,
        conflict=(0.0, 0.0),
        G=0.0,
        F=F,
        K=sum(i.length for i in interventions),
        J=0.0,
    )
    fields.update(overrides)
    return RepairResult(**fields)


def test_package_exports_core_types():
    for name in poc.__all__:
        assert getattr(poc, name) is getattr(poc.types, name)


def test_every_intervention_kind_is_exactly_one_of_diagnostic_or_executable():
    for kind in InterventionKind:
        assert kind.is_diagnostic != kind.is_executable


def test_diagnostic_and_executable_kinds_are_both_present():
    assert {k for k in InterventionKind if k.is_diagnostic} == {
        InterventionKind.REMOVE,
        InterventionKind.DISABLE_COLLISION,
    }
    assert {k for k in InterventionKind if k.is_executable} == {
        InterventionKind.RELOCATE,
        InterventionKind.SHIFT_TARGET,
    }


def test_repair_result_rejects_diagnostic_interventions():
    remove_wall = Intervention("d_wall", InterventionKind.REMOVE, "wall")
    with pytest.raises(ValueError, match="not executable"):
        _result((remove_wall,))


def test_records_are_immutable_so_state_cannot_be_corrupted_across_subsets():
    action = ActionSpec("insert", "cup", params=(0.0, 1.0))
    ip = _relocate("I1", "box")
    for record in (action, ip, Entity("box", EntityRole.MOVABLE)):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, next(iter(dataclasses.asdict(record))), None)


def test_selected_set_is_order_independent():
    i1, i2, i3 = _relocate("I1", "a"), _relocate("I2", "b"), _relocate("I3", "c")
    r_fwd = _result((i1, i2, i3))
    r_rev = _result((i3, i1, i2))
    assert r_fwd.selected_ids == r_rev.selected_ids == {"I1", "I2", "I3"}
    assert len({i1, i2, i3, _relocate("I1", "a")}) == 3  # hashable, value equality


def test_repair_set_has_set_semantics():
    i1 = _relocate("I1", "a")
    with pytest.raises(ValueError, match="duplicate"):
        _result((i1, i1))


def test_K_counts_explicit_macro_lengths():
    s = (_relocate("I1", "a", length=1), _relocate("I2", "b", length=3))
    assert _result(s).K == 4
    with pytest.raises(ValueError, match="does not match"):
        _result(s, K=2)


def test_empty_set_has_zero_repair_length():
    r = _result((), F=1, G=2.5, conflict=(2.5, 0.0))
    assert r.K == 0 and not r.feasible


@pytest.mark.parametrize("bad_length", [0, -1, 1.5, True])
def test_intervention_length_must_be_positive_integer(bad_length):
    with pytest.raises(ValueError, match="length"):
        _relocate("I1", "a", length=bad_length)


@pytest.mark.parametrize("bad_F", [-1, 2, 0.5])
def test_feasibility_is_binary(bad_F):
    with pytest.raises(ValueError, match="Feasibility"):
        _result((), F=bad_F)


def test_conflict_and_graded_conflict_are_nonnegative():
    with pytest.raises(ValueError, match="c_i"):
        _result((), conflict=(0.1, -0.2))
    with pytest.raises(ValueError, match="Graded"):
        _result((), G=-1.0)


def test_structural_entities_are_not_movable():
    assert not Entity("wall", EntityRole.STRUCTURAL).movable
    assert Entity("box", EntityRole.MOVABLE).movable
    assert Entity("cup", EntityRole.TARGET).movable


def test_empty_identifiers_are_rejected():
    with pytest.raises(ValueError):
        ActionSpec("", "cup")
    with pytest.raises(ValueError):
        Intervention("I1", InterventionKind.RELOCATE, "")
