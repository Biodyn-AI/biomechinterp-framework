PAPERDIR = paper
PAPER = main_v4
SUPP = supplementary_v4

.PHONY: all clean pdf supp

all: pdf supp

pdf:
	cd $(PAPERDIR) && pdflatex $(PAPER) && bibtex $(PAPER) && pdflatex $(PAPER) && pdflatex $(PAPER)

supp:
	cd $(PAPERDIR) && pdflatex $(SUPP) && bibtex $(SUPP) && pdflatex $(SUPP) && pdflatex $(SUPP)

clean:
	cd $(PAPERDIR) && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls

distclean: clean
	cd $(PAPERDIR) && rm -f $(PAPER).pdf $(SUPP).pdf

view:
	open $(PAPERDIR)/$(PAPER).pdf
