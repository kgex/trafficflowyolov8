# Function to display error message with line number
print_error() {
    echo "Error occurred on line $1"
}


# Trap any ERR signals and call print_error function
trap 'print_error $LINENO' ERR

set -e

docker compose up &
sleep 1m &
cd trafficflowyolov8;
source venv/bin/activate; 
python3 ./scripts/main.py --input ./input/final_input.MOV 

