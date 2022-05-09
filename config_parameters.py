import yaml

basic_config_file_path = "configs.yml"

class Config():
    def __init__(self, conf_file=None):
        self.config_file = config_file
        if conf_file:
            self.configs = yaml.safe_load(open(self.config_file))
        
    def load_conf(self, config_file_path):
        self.configs = yaml.safe_load(open(config_file_path))
        