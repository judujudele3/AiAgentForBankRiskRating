# utils/tool_utils.py

def format_success(outputs: dict, logs: list = []) -> dict:
    return {
        "status": "success",
        "outputs": outputs,
        "logs": logs
    }

def format_failure(error_msg: str) -> dict:
    return {
        "status": "error",
        "message": error_msg
    }

def validate_required_keys(inputs: dict, required: list):
    missing = [key for key in required if key not in inputs]
    if missing:
        return format_failure(f"Missing required input(s): {missing}")
    return None
