SRC= main.py data.py model.py main_draw.py
REST= README makefile 


#interactive
fit: 	main.py
	pypy main.py
draw:	main_draw.py
	python main_draw.py 

clean:
	\rm -r -f *.pyc output* *~ all.tar


all.tar: $(SRC) $(REST)
	tar cvf all.tar $(SRC) $(REST)
