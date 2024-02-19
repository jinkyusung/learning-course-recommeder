import time


def performance_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"\033[1;32m[Start]\033[0m {func.__name__} is starting...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\033[1;32m[Done]\033[0m {func.__name__} executed in: \033[1;34m{format_time(elapsed_time)}\033[0m\n")
        return result
    return wrapper


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"


def display_leader_board(item_scores, title="Leader Board"):
    very_peri_color = "\033[38;2;140;102;196m"
    reset_color = "\033[0m"
    print(f"{very_peri_color}{title:^37}{reset_color}")
    header = "| {:^6} | {:^8} | {:^13} |".format("Rank", "Course", "Score")
    line = "+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 15 + "+"
    print(line)
    print(header)
    print(line)
    for rank, (score, course) in enumerate(item_scores, start=1):
        print("| {:^6} | {:^8} | {:^13.5f} |".format(f"#{rank}", int(course), score))
    print(line)
    print()


def display_courses(recommended: list) -> None:
    color_code = '\033[38;2;140;102;196m'
    reset_code = '\033[0m'
    formatted_subject = f"{color_code}Top{len(recommended)} Recommended Courses{reset_code}"
    print(formatted_subject)
    for order, course in enumerate(recommended, 1):
        print("{:4} {}".format(f"#{order}", course))
    print()
