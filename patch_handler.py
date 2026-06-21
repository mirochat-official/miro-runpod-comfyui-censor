from pathlib import Path


handler_path = Path("/handler.py")

if not handler_path.exists():
    raise RuntimeError("/handler.py not found")

text = handler_path.read_text()

if "from censor import censor_image_bytes" not in text:
    if "import logging\n" in text:
        text = text.replace(
            "import logging\n",
            "import logging\nfrom censor import censor_image_bytes\n",
            1,
        )
    elif "import traceback\n" in text:
        text = text.replace(
            "import traceback\n",
            "import traceback\nfrom censor import censor_image_bytes\n",
            1,
        )
    else:
        raise RuntimeError("Could not find import location in /handler.py")

if "miro-censor - image censor pass completed" not in text:
    lines = text.splitlines(keepends=True)
    new_lines = []
    inserted = False

    target = "image_bytes = get_image_data(filename, subfolder, img_type)"

    for line in lines:
        new_lines.append(line)

        if line.strip() == target:
            indent = line[: len(line) - len(line.lstrip())]

            new_lines.extend(
                [
                    f"{indent}if image_bytes:\n",
                    f"{indent}    try:\n",
                    f"{indent}        image_bytes = censor_image_bytes(image_bytes)\n",
                    f"{indent}        print(\"miro-censor - image censor pass completed\")\n",
                    f"{indent}    except Exception as censor_error:\n",
                    f"{indent}        print(f\"miro-censor - censor failed: {{censor_error}}\")\n",
                    f"{indent}        if os.environ.get(\"CENSOR_FAIL_OPEN\", \"false\").lower() == \"true\":\n",
                    f"{indent}            print(\"miro-censor - CENSOR_FAIL_OPEN=true, returning original image\")\n",
                    f"{indent}        else:\n",
                    f"{indent}            raise\n",
                ]
            )

            inserted = True

    if not inserted:
        raise RuntimeError("Could not find image_bytes insertion point in /handler.py")

    text = "".join(new_lines)

handler_path.write_text(text)

print("miro-censor - /handler.py patched successfully")
