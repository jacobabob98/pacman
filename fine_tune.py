import os
import subprocess

K_vals = list(range(300, 900, 100))
gamma_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

layout = 'layouts/q1a_tinyMaze.lay'
agent = 'Q1Agent'
ghost = 'StationaryGhost'
n_runs = 10

max_avg_score = -float('inf')
best_K = None
best_gamma = None

for K in K_vals:
    for gamma in gamma_vals:
        cmd = f"python pacman.py -l {layout} -p {agent} -a discount={gamma},iterations={K} -g {ghost} -n {n_runs}"
        
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = proc.stdout
        
        lines = output.split('\n')
        for line in reversed(lines):
            if 'Average Score:' in line:
                words = line.split()
                index = words.index('Score:')
                average_score = float(words[index + 1])
                break
        
        if average_score > max_avg_score:
            max_avg_score = average_score
            best_K = K
            best_gamma = gamma
            
print(f"Best K: {best_K}, Best gamma: {best_gamma}, Maximum Average Score: {max_avg_score}")