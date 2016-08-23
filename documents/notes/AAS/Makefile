SUFFIX   = pdf aux log dvi ps out

NAME = Abstract

all: ${NAME}.pdf

%.pdf: %.tex
	pdflatex $<

clean:
	rm -rf ${foreach suff, ${SUFFIX}, ${NAME}.${suff}}