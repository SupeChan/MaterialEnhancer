import dataclasses

@dataclasses.dataclass
class ScaleMethod:
    X1 = "scale1x"
    X2 = "scale2x"
    X4 = "scale4x"


DIC_METHOD_TO_SIZE={
    ScaleMethod.X1:"x1",
    ScaleMethod.X2:"x2",
    ScaleMethod.X4:"x4",
}
