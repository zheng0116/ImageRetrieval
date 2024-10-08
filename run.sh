case $1 in
"start")
    python3 main.py --model_path ./Dinov2_model/dinov2-small --model_size small --database_folder ./quary
    ;;
"install")
    python3 -m venv .
    source ./bin/activate
    pip3 install -r requirements.txt
esac