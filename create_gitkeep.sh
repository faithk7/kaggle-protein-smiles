#!/bin/bash

# Directory containing the folders
root_dir="."

# Check if the directory exists
if [ ! -d "$root_dir" ]; then
    echo "Directory $root_dir does not exist."
    exit 1
fi

# Function to create .gitkeep in all directories and subdirectories
create_gitkeep() {
    for dir in "$1"/*; do
        if [ -d "$dir" ]; then
            # Create .gitkeep file if it doesn't exist
            touch "$dir/.gitkeep"
            echo "Created .gitkeep in $dir"

            # Recursively call the function for subdirectories
            create_gitkeep "$dir"
        fi
    done
}

# Start the process - here we call the function defined above
create_gitkeep "$root_dir"

echo "All .gitkeep files created."

