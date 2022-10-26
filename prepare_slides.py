import nbformat as nbf
import argparse

def prepare_for_presentation(ipynb_file):
    ntbk = nbf.read(ipynb_file, nbf.NO_CONVERT)
    for c in ntbk.cells:
        #c["metadata"] = {"slideshow": {"slide_type": "slide"}, "tags": ["skiprun"]}
        c["metadata"] = {"tags": ["skiprun"]}
    nbf.write(ntbk, ipynb_file, version=nbf.NO_CONVERT)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help="notebook to prepare for presentation")
    args = parser.parse_args()
    prepare_for_presentation(args.filename)