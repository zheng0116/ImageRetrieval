case $1 in
"start")
    python3 main.py 
    ;;
"install")
    python3 -m venv .
    source ./bin/activate
    pip3 install -r requirements.txt
esac
