import ultralytics.nn.tasks as tasks
from src.models.c2f_swin import C2f_Swin


def register_custom_modules() -> None:
    tasks.__dict__["C2f_Swin"] = C2f_Swin
    setattr(tasks, "C2f_Swin", C2f_Swin)