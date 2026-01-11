import sys

class RAGException(Exception):

    def __init__(self, errormessage, errordetials:sys):
        self.errormessage = errormessage
        _ , _ , exc_tb = errordetials.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python scripts name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.errormessage)
        )
