


class Metrics():
    """
    Definisce un generico oggetto che contiene metriche generiche e Embedding
    """

    def __init__(self, **kwargs):
        self.attributi = {}
        for k,v in kwargs.items():
            self.attributi[k] = v

    def get_metric(self, string_name_of_value):
        if string_name_of_value in self.attributi.keys():
            return self.attributi[string_name_of_value]

    #def add_metric(self):