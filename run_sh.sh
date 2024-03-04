# Function to display error message with line number
print_error() {
    echo "Error occurred on line $1"
}

while getopts h:a:i: flag
do
    case "${flag}" in
        h) host=${OPTARG};;
        a) token=${OPTARG};;
        i) input=${OPTARG};;
    esac

# Trap any ERR signals and call print_error function
trap 'print_error $LINENO' ERR

set -e

docker compose up -d &
sleep 1m;
source venv/bin/activate; 
python3 ./scripts/main.py --input rtsp://admin:Kgisl@123@172.50.16.235:554/Streaming/Channels/101

