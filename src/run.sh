for i in {1..10}
do
	echo $i
	python3 shortest_path.py --epochs=5 --lr=0.0001
done
