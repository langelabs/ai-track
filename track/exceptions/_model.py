class ModelNotFound(Exception):
    def __init__(self, model:str):
        self.add_note(f"Model {model} not found.")
