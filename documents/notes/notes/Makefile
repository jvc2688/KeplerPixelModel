SUFFIX   = pdf aux log dvi ps out

NAME = regularization Auto-Regression

all: ${foreach name, ${NAME}, ${name}.pdf}

%.pdf: %.tex
	pdflatex $<

clean:
	rm -rf ${foreach name, ${NAME}, ${foreach suff, ${SUFFIX}, ${name}.${suff}}}