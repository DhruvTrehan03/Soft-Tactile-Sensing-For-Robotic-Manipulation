import matplotlib.pyplot as plt
import numpy as np

# Define the heads and their error percentages
heads = ['Shape', 'Position', 'Orientation']
error_percentages = [12.5, 18.3, 9.7]  # Replace with your actual error percentages

# X-axis positions for the bars
x = np.arange(len(heads))
width = 0.5

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, error_percentages, width, color=['skyblue', 'salmon', 'lightgreen'])

# Add labels above each bar indicating the error percentage
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', 
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Set labels and title
ax.set_ylabel('Error Percentage (%)')
ax.set_title('Error Percentage for Each Head')
ax.set_xticks(x)
ax.set_xticklabels(heads)

plt.tight_layout()
plt.show()