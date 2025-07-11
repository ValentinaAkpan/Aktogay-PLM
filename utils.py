# Currently empty, but can include helper functions like format_hour if needed
def format_hour(hour):
    """Format hour for display (e.g., 13 -> 1 PM)."""
    return f"{hour%12 if hour != 12 else 12} {'AM' if hour < 12 else 'PM'}"
