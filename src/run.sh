for i in {1..10}
do
	echo $i
	python3 shortest_path.py --epochs=10 --lr=0.00002
done
