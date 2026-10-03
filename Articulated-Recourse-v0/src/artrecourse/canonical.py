"""Hand-designed canonical scenes, one per core variant (V0-V7).

Each spec only fixes WHAT is in the scene (assets, exact initial poses, candidate
interventions). Whether it really is V_k is decided by the oracle on the resulting
MuJoCo geometry (variants.check_variant); the physical story is documentation.

Recurring geometry (pulled-out upper rack, world frame):
  * a skillet resting on the tines with its handle pointing out of the rack front
    (yaw -90) protrudes ~9 cm past the rack: once the rack is pushed in, the closing
    door hits the handle;
  * a 0.20 m water bottle standing in a rack bay is taller than the 0.187 m clearance
    under the tub ceiling: pushing the rack in jams it against the ceiling;
  * relocations to the counter travel with the object's bottom 13 cm above the counter
    (just above the rack walls), so tall items standing on the counter between the rack
    and a far counter slot block the transfer corridor (destination may be free).
"""

from __future__ import annotations

from .interventions import InterventionSpec as I
from .interventions import ObjSpec as O
from .interventions import SceneSpec

PAN_OUT = dict(slot="upper_center_back", yaw_deg=-90, dxy=(0.0, -0.005))   # handle sticks out of the rack front
BOX_IN_CORRIDOR = dict(slot="counter_buffer_right", yaw_deg=0, dxy=(-0.07, -0.055))  # box at (0.70, -0.26)


def v0() -> SceneSpec:
    return SceneSpec(
        "canonical_V0", split="canonical", variant_requested="V0",
        story="The skillet's handle sticks out of the front of the pulled upper rack. Once the rack is pushed "
              "in, the closing door hits the handle. Turning the skillet so the handle lies across the rack "
              "(one relocation) makes CLOSE_DISHWASHER feasible.",
        objects=[
            O("pan", "pan/pan_2", **PAN_OUT),
            O("mug", "mug/mug_1", "counter_buffer_left", 90),
            O("cup", "cup/cup_3", "counter_buffer_center", 0),
            O("can", "can/can_2", "counter_left_near", 0),
            O("bowl", "bowl/bowl_2", "counter_buffer_far_right", 0),
        ],
        interventions=[
            I("pan", "upper_tines_handle_left", 180),
            I("pan", "upper_tines_handle_right", 0),
            I("mug", "counter_buffer_right", 90),
            I("can", "counter_left_mid", 0),
        ])


def v1() -> SceneSpec:
    return SceneSpec(
        "canonical_V1", split="canonical", variant_requested="V1",
        story="Two independent blockers: the skillet handle protrudes into the door's sweep and a 0.20 m water "
              "bottle in the right bay would jam against the tub ceiling while the rack is pushed in. Both "
              "must move, in either order; neither relocation affects the other.",
        objects=[
            O("pan", "pan/pan_2", **PAN_OUT),
            O("bottle", "bottle/water_bottle_1", "upper_right_front", 0, (0.016, -0.02)),
            O("mug", "mug/mug_1", "counter_buffer_left", 90),
            O("can", "can/can_2", "counter_buffer_far_right", 0),
        ],
        interventions=[
            I("pan", "upper_tines_handle_left", 180),
            I("bottle", "counter_buffer_center", 0),
            I("bottle", "counter_buffer_right", 0),
            I("mug", "counter_left_near", 90),
        ])


def v2() -> SceneSpec:
    return SceneSpec(
        "canonical_V2", split="canonical", variant_requested="V2",
        story="The water bottle in the right bay jams the rack. Both reachable buffer poses for it are occupied "
              "(a bowl at the counter centre, a can at the counter right). One occupant must be moved aside "
              "first: destination occupancy creates the dependency.",
        objects=[
            O("bottle", "bottle/water_bottle_1", "upper_right_back", 0),
            O("bowl", "bowl/bowl_2", "counter_buffer_center", 0),
            O("can", "can/can_2", "counter_buffer_right", 0),
            O("mug", "mug/mug_1", "upper_left_front", 90),
        ],
        interventions=[
            I("bottle", "counter_buffer_center", 0),
            I("bottle", "counter_buffer_right", 0),
            I("bowl", "counter_buffer_left", 0),
            I("can", "counter_buffer_far_right", 0),
            I("mug", "counter_left_near", 90),
        ])


def v3() -> SceneSpec:
    return SceneSpec(
        "canonical_V3", split="canonical", variant_requested="V3",
        story="The water bottle jams the rack. Its buffer poses at the right end of the counter are FREE, but "
              "the bottle is carried 13 cm above the counter and its transfer corridor passes through a tall "
              "cereal box. Moving the box out of the corridor makes the exact bottle relocation executable: a "
              "swept-volume dependency, not an occupied destination.",
        objects=[
            O("bottle", "bottle/water_bottle_1", "upper_right_back", 0),
            O("box", "box/boxed_food_2", **BOX_IN_CORRIDOR),
            O("mug", "mug/mug_1", "upper_left_front", 90),
            O("can", "can/can_2", "counter_left_near", 0),
        ],
        interventions=[
            I("bottle", "counter_buffer_far_right", 0),
            I("bottle", "counter_buffer_right", 0),
            I("box", "counter_buffer_left", 0),
            I("box", "counter_left_mid", 0),
            I("mug", "counter_buffer_center", 90),
        ])


def v4() -> SceneSpec:
    return SceneSpec(
        "canonical_V4", split="canonical", variant_requested="V4",
        story="Two independent dependency chains in one scene. Right: a cereal box stands in the corridor of "
              "the water bottle (right bay) -> box, then bottle. Left (mirror image): a second box stands in "
              "the corridor of the tall bottle (left bay) -> box, then bottle. Both bottles jam the rack.",
        objects=[
            O("bottle_r", "bottle/water_bottle_1", "upper_right_back", 0),
            O("box_r", "box/boxed_food_2", **BOX_IN_CORRIDOR),
            O("bottle_l", "bottle/water_bottle_3", "upper_left_back", 0),
            O("box_l", "box/boxed_food_6", "counter_left_near", 0, (-0.10, -0.075)),
            O("mug", "mug/mug_1", "upper_right_front", 90),
        ],
        interventions=[
            I("bottle_r", "counter_buffer_far_right", 0),
            I("bottle_r", "counter_buffer_right", 0),
            I("box_r", "counter_buffer_left", 0),
            I("bottle_l", "counter_left_far", 0),
            I("bottle_l", "counter_left_mid", 0),
            I("box_l", "counter_left_near", 0),
            I("mug", "counter_buffer_far_right", 90),
        ])


def v5() -> SceneSpec:
    return SceneSpec(
        "canonical_V5", split="canonical", variant_requested="V5",
        story="A clean three-object chain. A tall bottle standing right behind the cereal box blocks the "
              "top-down grasp of the box (the hand would hit it). The box stands in the transfer corridor of "
              "the water bottle, and the water bottle jams the rack. Bottle(counter) -> box -> water bottle -> "
              "close.",
        objects=[
            O("bottle_a", "bottle/water_bottle_1", "upper_right_back", 0),
            O("box", "box/boxed_food_2", **BOX_IN_CORRIDOR),
            O("bottle_c", "bottle/water_bottle_0", "counter_buffer_right", 0, (-0.07, 0.04)),
            O("mug", "mug/mug_1", "upper_left_front", 90),
        ],
        interventions=[
            I("bottle_a", "counter_buffer_far_right", 0),
            I("bottle_a", "counter_buffer_right", 0),
            I("box", "counter_buffer_center", 0),
            I("box", "counter_left_mid", 0),
            I("bottle_c", "counter_buffer_left", 0),
            I("bottle_c", "counter_left_near", 0),
            I("mug", "counter_left_far", 90),
        ])


def v6() -> SceneSpec:
    return SceneSpec(
        "canonical_V6", split="canonical", variant_requested="V6",
        story="Branching recourse. Path A (2 moves): the bowl occupying the counter centre is moved aside, then "
              "the water bottle goes to the centre. Path B (3 moves): the tall bottle behind the cereal box is "
              "moved so the box can be grasped, the box leaves the corridor, then the water bottle goes to the "
              "counter right. The oracle must prefer the shorter coordinated plan.",
        objects=[
            O("bottle_a", "bottle/water_bottle_1", "upper_right_back", 0),
            O("bowl", "bowl/bowl_2", "counter_buffer_center", 0, (-0.03, 0.0)),
            O("box", "box/boxed_food_2", "counter_buffer_right", 0, (-0.03, -0.065)),
            O("bottle_c", "bottle/water_bottle_0", "counter_buffer_right", 0, (-0.03, 0.04)),
            O("mug", "mug/mug_1", "upper_left_front", 90),
        ],
        interventions=[
            I("bottle_a", "counter_buffer_center", 0),
            I("bottle_a", "counter_buffer_right", 0),
            I("bowl", "counter_buffer_left", 0),
            I("box", "counter_buffer_far_right", 0),
            I("bottle_c", "counter_buffer_left", 0, (-0.04, 0.03)),
            I("mug", "counter_left_far", 90),
        ])


def v7() -> SceneSpec:
    return SceneSpec(
        "canonical_V7", split="canonical", variant_requested="V7",
        story="Dense scene, three objects block CLOSE_DISHWASHER: the skillet handle (door), a 0.20 m water bottle "
              "(right bay) and a 0.25 m bottle (left bay) (tub ceiling). The cereal box on the counter stands in "
              "the transfer corridors of BOTH bottles; the tall bottle also stands under the skillet's "
              "handle-across pose and in its swing. Minimal recourse: box -> water bottle -> tall bottle -> "
              "skillet (or the skillet turned right once the water bottle is gone).",
        objects=[
            O("pan", "pan/pan_2", **PAN_OUT),
            O("bottle_r", "bottle/water_bottle_1", "upper_right_back", 0),
            O("bottle_l", "bottle/water_bottle_3", "upper_left_back", 0),
            O("box", "box/boxed_food_2", **BOX_IN_CORRIDOR),
            O("mug", "mug/mug_1", "counter_left_far", 90, (-0.04, 0.0)),
            O("cup", "cup/cup_3", "counter_left_near", 0),
            O("bowl", "bowl/bowl_2", "counter_buffer_far_right", 0),
            O("can", "can/can_2", "counter_buffer_center", 0, (0.0, 0.025)),
        ],
        interventions=[
            I("pan", "upper_tines_handle_left", 180),
            I("pan", "upper_tines_handle_right", 0),
            I("bottle_r", "counter_buffer_far_right", 0),
            I("bottle_r", "counter_buffer_right", 0),
            I("bottle_l", "counter_buffer_right", 0),
            I("bottle_l", "counter_left_mid", 0),
            I("box", "counter_buffer_left", 0),
            I("box", "counter_left_mid", 0),
            I("mug", "counter_buffer_center", 90),
            I("mug", "counter_left_mid", 90),
            I("cup", "upper_right_back", 0),
            I("bowl", "counter_buffer_left", 0),
            I("can", "counter_buffer_far_right", 0),
        ])


ALL = {"V0": v0, "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6, "V7": v7}
