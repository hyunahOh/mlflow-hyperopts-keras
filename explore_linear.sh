for alpha in 0 0.2 0.5 0.7 ; do
    for l1 in 0.2 0.5 0.7 ; do
        echo "==================================="
        echo "python3 train.py $alpha $l1"        
        echo "==================================="
        python3 train.py $alpha $l1
    done
done
