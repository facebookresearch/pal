#!/usr/bin/bash

# Sample of `plot_config`:
# plot_config = {
#     "grid_size": [4, 4],
#     "plots": [
#         {"type": "show_token_emb", "position": [0, 0]},
#         {"type": "show_pos_emb", "position": [0, 1]},
#         {"type": "show_emb", "position": [0, 2]},
#         {"type": "show_norm_emb", "position": [0, 3]},
#         {"type": "show_attn", "position": [1, 0]},
#         {"type": "show_value", "position": [1, 1]},
#         {"type": "show_seq_emb", "position": [1, 2]},
#         {"type": "show_level_line", "position": [1, 3]},
#         {"type": "show_norm_input", "position": [2, 0]},
#         {"type": "show_mlp_receptors", "position": [2, 1]},
#         {"type": "show_mlp_emitters", "position": [2, 2]},
#         {"type": "show_mlp_output", "position": [2, 3]},
#         {"type": "show_output_level_lines", "position": [3, 0]},
#         {"type": "show_output", "position": [3, 1]},
#         {"type": "show_loss", "position": [3, 2]},
#         {"type": "show_acc", "position": [3, 3]},
#     ],
# }

while read line; do
  name=$(echo $line | cut -d ' ' -f 1)
  timestamp=$(echo $line | cut -d ' ' -f 2)
  # Multiply by 20 to get the epoch number
  epoch=$(echo $timestamp | awk '{print $1 * 20}')
  python ../visualization.py frame --epoch ${epoch%.*} --save-ext seed --unique-id $name --plot_config '{\
            "grid_size": [1, 2], \
            "plots": [ \
                {"type": "show_pos_emb", "position": [0]}, \
                {"type": "show_norm_emb", "position": [1]}, \
            ], \
        }'
done < $1