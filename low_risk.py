
def calculate_average(numbers):
    """Calculate average of numbers"""
    return sum(numbers) / len(numbers)

def main():
    data = [1, 2, 3, 4, 5]
    result = calculate_average(data)
    print(f"Average: {result}")

if __name__ == "__main__":
    main()