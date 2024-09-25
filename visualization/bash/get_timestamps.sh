#!/usr/bin/bash

while read line; do
  name=$(echo $line | cut -d ' ' -f 1)
  timestamp=$(echo $line | cut -d ' ' -f 2)
  # Multiply by 20 to get the epoch number
  epoch=$(echo $timestamp | awk '{print $1 * 20}')
  python ../visualization.py frame --epoch ${$(($timestamp * 20))%.*} --file-format png --save-ext test --unique-id $name
done < "$1"