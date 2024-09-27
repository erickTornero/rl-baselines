from omegaconf import OmegaConf
import os
from datetime import datetime
import uuid

class SaveUtils:
    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg

    def allocate_saving_folders(self):
        base_path = self.cfg.exp_root_dir
        project_datetime = datetime.now().strftime("%d-%m-%y-%Hh%Mm%Ss")
        project_uuid = "".join((str(uuid.uuid4())).split("-"))[:8]
        project_name = f"{project_datetime}-{project_uuid}"
        project_folder = os.path.join(base_path, project_name)
        if os.path.exists(project_folder):
            raise FileExistsError("Project exists")
        self.cfg.project_folder = project_folder
        os.makedirs(self.cfg.project_folder)
        os.makedirs(os.path.join(self.cfg.project_folder, "val_predictions"))
        self.save_cfg("parsed.yaml")
        print("Project Folder -> ", project_folder)

    def get_absolute_path(self, name_file: str):
        return os.path.join(self.cfg.project_folder, name_file)

    #TODO: val predictions hardcoded
    def save_text(self, name_file: str, text: str):
        path = self.get_absolute_path(os.path.join("val_predictions", name_file))
        with open(path, "w") as fp:
            fp.write(text)

    def save_cfg(self, name_file: str):
        path = self.get_absolute_path(name_file)
        OmegaConf.save(self.cfg, path)