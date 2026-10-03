"""Hand-designed canonical scenes V0-V7 (v0.1).

Every object pose is an exact named slot + named orientation template (topology.py); there
are NO free offsets. Target in every scene: the atomic PUSH_RACK of the upper rack (pulled to
its loading position under the open door). Whether a scene is V_k is decided by the oracle.

Physical vocabulary (all measured, see out/fixture_audit.md):
  * tub-ceiling jam: an item taller than the 0.187 m clearance between the upper rack floor and
    the tub ceiling (0.20 m water bottle, 0.175 m utensil holder on the 77 mm tine plates);
  * push-hand conflict: a skillet with its handle out over the front wall sits exactly where the
    robot's hand grasps the wall to push the rack;
  * transfer corridor: a relocation lifts the object just over every lip it crosses (rack walls,
    countertop edge, tray rim) + 3 cm. Leaving the rack for the tray's back row therefore passes
    low over the front row: anything standing there blocks the carried object;
  * occupancy: a capacity-1 slot holding another object.
"""

from __future__ import annotations

from .interventions import InterventionSpec as I
from .interventions import ObjSpec as O
from .interventions import SceneSpec

HOLDER = "utensil_holder/BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028"


def v0() -> SceneSpec:
    return SceneSpec(
        "canonical_V0", variant_requested="V0",
        story="A porcelain utensil holder stands on the upper rack's tine field. On top of the tines it is taller "
              "than the tub opening, so pushing the rack in jams it against the tub ceiling. One relocation onto "
              "the drying tray's free first column fixes it.",
        objects=[O("holder", HOLDER, "U5"), O("cup", "cup/cup_3", "U1"), O("can", "can/can_2", "B2"),
                 O("mug", "mug/mug_1", "B3"), O("bowl", "bowl/bowl_2", "BW4")],
        interventions=[I("holder", "BW1"), I("holder", "BW2"), I("cup", "B7"), I("mug", "B7")])


def v1() -> SceneSpec:
    return SceneSpec(
        "canonical_V1", variant_requested="V1",
        story="Two independent blockers, both too tall for the tub: a 0.20 m water bottle in the right bay and a "
              "porcelain utensil holder on the tine field. Each goes to its own tray cell; neither move affects the "
              "other, so either order works.",
        objects=[O("bottle", "bottle/water_bottle_1", "U3"), O("holder", HOLDER, "U5"),
                 O("cup", "cup/cup_3", "B8"), O("bowl", "bowl/bowl_0", "L1")],
        interventions=[I("bottle", "B3"), I("holder", "BW1"), I("cup", "U2")])


def v2() -> SceneSpec:
    return SceneSpec(
        "canonical_V2", variant_requested="V2",
        story="The water bottle must leave the rack, but the tray's front row is full: a cup on B1, a cereal box on "
              "B2, a can on B3. The bottle's buffer cell B1 is visibly OCCUPIED by the cup; loading the cup into the "
              "rack frees exactly that cell. (Cell B3 would need the can moved and is behind the box anyway.)",
        objects=[O("bottle", "bottle/water_bottle_1", "U3"), O("cup", "cup/cup_3", "B1"),
                 O("box", "box/boxed_food_2", "B2"), O("can", "can/can_2", "B3")],
        interventions=[I("bottle", "B1"), I("bottle", "B3"), I("cup", "U1"), I("cup", "U2"), I("can", "B7")])


def v3() -> SceneSpec:
    return SceneSpec(
        "canonical_V3", variant_requested="V3",
        story="The water bottle must leave the rack. The back-row cells B5 and B6 are visibly EMPTY, but the bottle "
              "is carried just over the tray rim and its path into the back row crosses the front-left cell B1, "
              "where a cup stands. Loading the cup into the rack clears the path; the SAME bottle relocation to the "
              "SAME empty cell then succeeds. No destination is occupied.",
        objects=[O("bottle", "bottle/water_bottle_1", "U3"), O("cup", "cup/cup_3", "B1"),
                 O("box", "box/boxed_food_2", "B4")],
        interventions=[I("bottle", "B6"), I("bottle", "B5"), I("cup", "U1"), I("cup", "U2"), I("box", "B8")])


def v4() -> SceneSpec:
    return SceneSpec(
        "canonical_V4", variant_requested="V4",
        story="Two different dependency chains. Swept volume: the water bottle's path to the back cell B7 crosses the "
              "cereal box on B2, which is slid to B3 first. Occupancy: the utensil holder (too tall for the tub) "
              "needs the tray's first column, where a cup stands on B5; the cup is loaded into the rack first.",
        objects=[O("bottle", "bottle/water_bottle_1", "U3"), O("box", "box/boxed_food_2", "B2"),
                 O("holder", HOLDER, "U5"), O("cup", "cup/cup_3", "B5")],
        interventions=[I("bottle", "B7"), I("box", "B3"), I("holder", "BW1"), I("cup", "U1")])


def v5() -> SceneSpec:
    return SceneSpec(
        "canonical_V5", variant_requested="V5",
        story="A three-object chain where the blocker is NOT the first thing the robot can move. The water bottle "
              "(tub-ceiling jam) must reach the back cell B7; its path crosses the cup on B2. The cup's own path into "
              "the rack crosses the can on B1. So: can to the back row, cup into the rack, bottle onto the tray, then "
              "push the rack.",
        objects=[O("bottle", "bottle/water_bottle_1", "U3"), O("cup", "cup/cup_3", "B2"),
                 O("can", "can/can_2", "B1")],
        interventions=[I("bottle", "B7"), I("bottle", "B3"), I("cup", "U1"), I("can", "B5")])


def v6() -> SceneSpec:
    return SceneSpec(
        "canonical_V6", variant_requested="V6",
        story="Two ways to clear the water bottle in the left bay. Short (occupancy, 2 moves): load the cup standing "
              "on back cell B5 into the rack and put the bottle there. Long (swept volume, 3 moves): the free front "
              "cell B3 is reached over the cereal box on B2; the box can only slide back to B6, where a can stands, "
              "so the can moves to B7 first.",
        objects=[O("bottle", "bottle/water_bottle_1", "U1"), O("cup", "cup/cup_3", "B5"),
                 O("box", "box/boxed_food_2", "B2"), O("can", "can/can_2", "B6")],
        interventions=[I("bottle", "B5"), I("bottle", "B3"), I("cup", "U2"), I("box", "B6"), I("can", "B7")])


def v7() -> SceneSpec:
    return SceneSpec(
        "canonical_V7", variant_requested="V7",
        story="Dense scene, two blockers: the skillet's handle sits where the push-hand grasps the rack, and the "
              "water bottle in the left-front bay would jam the tub ceiling. Turning the skillet either way needs the "
              "bottle gone (its handle swings over it, or its pan body lands on it). The bottle's tray cells lie behind "
              "the cereal box on B2, and the box can only slide back to B6, where a can stands. Lower-rack bowls and "
              "cups are context.",
        objects=[O("pan", "pan/pan_2", "U5", "handle_out"), O("bottle", "bottle/water_bottle_1", "U1"),
                 O("box", "box/boxed_food_2", "B2"), O("can", "can/can_2", "B6"), O("mug", "mug/mug_1", "B4"),
                 O("cup", "cup/cup_3", "B8"), O("bowl", "bowl/bowl_0", "L1"), O("cup_l", "cup/cup_4", "L3")],
        interventions=[I("pan", "U5", "handle_left"), I("pan", "U5", "handle_right"),
                       I("bottle", "B3"), I("bottle", "B7"), I("bottle", "B8"), I("box", "B6"), I("box", "B5"),
                       I("can", "B5"), I("can", "B1"), I("mug", "B1"), I("mug", "B8"), I("cup", "U3")])


ALL = {"V0": v0, "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6, "V7": v7}
