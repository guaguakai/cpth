for i in {1..5}
do
	echo $i
	python3 shortest_path.py --epochs=10 --lr=0.0001
done
