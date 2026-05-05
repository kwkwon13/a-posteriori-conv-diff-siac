from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ARK3_2_4L_2_SA = "ARK3(2)4L[2]SA"
ARK5_4_8L_2_SA = "ARK5(4)8L[2]SA"


@dataclass(frozen=True, slots=True)
class IMEXRungeKuttaTableau:
    """Butcher tableau pair for IMEX Runge-Kutta time integration."""

    name: str
    order: int
    a_explicit: np.ndarray
    b_explicit: np.ndarray
    c_explicit: np.ndarray
    a_implicit: np.ndarray
    b_implicit: np.ndarray
    c_implicit: np.ndarray

    @property
    def num_stages(self) -> int:
        return int(self.a_explicit.shape[0])


def build_imex_runge_kutta_tableau(name: str) -> IMEXRungeKuttaTableau:
    if name == ARK3_2_4L_2_SA:
        return _build_ark3_tableau()
    if name == ARK5_4_8L_2_SA:
        return _build_ark5_tableau()
    raise ValueError(f"unsupported IMEX Runge-Kutta tableau: {name}")


def _build_ark3_tableau() -> IMEXRungeKuttaTableau:
    gamma = 0.4358665215
    c2 = _fraction(1767732205903, 2027836641118)
    c3 = 0.6
    a_explicit = _array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [c2, 0.0, 0.0, 0.0],
            [
                _fraction(5535828885825, 10492691773637),
                _fraction(788022342437, 10882634858940),
                0.0,
                0.0,
            ],
            [
                _fraction(6485989280629, 16251701735622),
                _fraction(-4246266847089, 9704473918619),
                _fraction(10755448449292, 10357097424841),
                0.0,
            ],
        ],
    )
    weights = _array(
        [
            _fraction(1471266399579, 7840856788654),
            _fraction(-4482444167858, 7529759066697),
            _fraction(11266239266428, 11593286722821),
            gamma,
        ],
    )
    c = _array([0.0, c2, c3, 1.0])
    a_implicit = _array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0],
            [
                _fraction(2746238789719, 10658868560708),
                _fraction(-640167445237, 6845629431997),
                gamma,
                0.0,
            ],
            [
                weights[0],
                weights[1],
                weights[2],
                gamma,
            ],
        ],
    )
    return IMEXRungeKuttaTableau(
        name=ARK3_2_4L_2_SA,
        order=3,
        a_explicit=a_explicit,
        b_explicit=weights,
        c_explicit=c,
        a_implicit=a_implicit,
        b_implicit=weights,
        c_implicit=c,
    )


def _build_ark5_tableau() -> IMEXRungeKuttaTableau:
    gamma = _fraction(41, 200)
    c2 = _fraction(41, 100)
    c3 = _fraction(2935347310677, 11292855782101)
    c4 = _fraction(1426016391358, 7196633302097)
    c5 = _fraction(92, 100)
    c6 = _fraction(24, 100)
    c7 = _fraction(3, 5)
    weights = _array(
        [
            _fraction(-872700587467, 9133579230613),
            0.0,
            0.0,
            _fraction(22348218063261, 9555858737531),
            _fraction(-1143369518992, 8141816002931),
            _fraction(-39379526789629, 19018526304540),
            _fraction(32727382324388, 42900044865799),
            gamma,
        ],
    )
    c = _array([0.0, c2, c3, c4, c5, c6, c7, 1.0])
    a_explicit = _array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [c2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                _fraction(367902744464, 2072280473677),
                _fraction(677623207551, 8224143866563),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                _fraction(1268023523408, 10340822734521),
                0.0,
                _fraction(1029933939417, 13636558850479),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                _fraction(14463281900351, 6315353703477),
                0.0,
                _fraction(66114435211212, 5879490589093),
                _fraction(-54053170152839, 4284798021562),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                _fraction(14090043504691, 34967701212078),
                0.0,
                _fraction(15191511035443, 11219624916014),
                _fraction(-18461159152457, 12425892160975),
                _fraction(-281667163811, 9011619295870),
                0.0,
                0.0,
                0.0,
            ],
            [
                _fraction(19230459214898, 13134317526959),
                0.0,
                _fraction(21275331358303, 2942455364971),
                _fraction(-38145345988419, 4862620318723),
                _fraction(-1, 8),
                _fraction(-1, 8),
                0.0,
                0.0,
            ],
            [
                _fraction(-19977161125411, 11928030595625),
                0.0,
                _fraction(-40795976796054, 6384907823539),
                _fraction(177454434618887, 12078138498510),
                _fraction(782672205425, 8267701900261),
                _fraction(-69563011059811, 9646580694205),
                _fraction(7356628210526, 4942186776405),
                0.0,
            ],
        ],
    )
    a_implicit = _array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                _fraction(41, 400),
                _fraction(-567603406766, 11931857230679),
                gamma,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                _fraction(683785636431, 9252920307686),
                0.0,
                _fraction(-110385047103, 1367015193373),
                gamma,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                _fraction(3016520224154, 10081342136671),
                0.0,
                _fraction(30586259806659, 12414158314087),
                _fraction(-22760509404356, 11113319521817),
                gamma,
                0.0,
                0.0,
                0.0,
            ],
            [
                _fraction(218866479029, 1489978393911),
                0.0,
                _fraction(638256894668, 5436446318841),
                _fraction(-1179710474555, 5321154724896),
                _fraction(-60928119172, 8023461067671),
                gamma,
                0.0,
                0.0,
            ],
            [
                _fraction(1020004230633, 5715676835656),
                0.0,
                _fraction(25762820946817, 25263940353407),
                _fraction(-2161375909145, 9755907335909),
                _fraction(-211217309593, 5846859502534),
                _fraction(-4269925059573, 7827059040749),
                gamma,
                0.0,
            ],
            weights,
        ],
    )
    return IMEXRungeKuttaTableau(
        name=ARK5_4_8L_2_SA,
        order=5,
        a_explicit=a_explicit,
        b_explicit=weights,
        c_explicit=c,
        a_implicit=a_implicit,
        b_implicit=weights,
        c_implicit=c,
    )


def _array(values: list | tuple) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _fraction(numerator: int, denominator: int) -> float:
    return float(np.float64(numerator / denominator))


__all__ = [
    "ARK3_2_4L_2_SA",
    "ARK5_4_8L_2_SA",
    "IMEXRungeKuttaTableau",
    "build_imex_runge_kutta_tableau",
]
