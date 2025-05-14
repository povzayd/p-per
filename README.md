# 🖼️ pcap2img

A command-line tool to extract all embedded JPEG images from any pcap file.  
Built with Python :)
---

## ✨ Features

- 📂 Automatically creates an output directory for extracted images
- 📸 Extracts all JPEG images found in the file (based on JPEG magic bytes)
- 🛡️ Handles file errors gracefully with clear messages

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install prompt_toolkit
```

### 2️⃣ Save the Script

Clone this repo:

```bash
git clone https://github.com/povzayd/pcap2img.git && cd pcap2img && python3 venv venv1 && source venv1/bin/activate && pip install prompt_toolkit && python pcap2img.py

```

---

## 🕹️ Usage

1. Run the script:

    ```bash
    python pcap2img.py
    ```

2. When prompted, enter the path to your binary file:

    ```
    🔍 Enter your file name: myfile.pcap
    ```

3. The tool will extract all JPEG images and save them in a new directory named after your file.

---

## 💡 Interactive CLI Powered by prompt_toolkit

- Enjoy advanced line editing, history, and emoji support in your prompts[1][5].
- Easily extend the prompt with right prompts, toolbars, or even auto-completion[1][11].

---

## 🤝 Contributing

Pull requests and suggestions welcome!  
Feel free to fork and enhance with more features (e.g., progress bars, file previews).


---

> Made with ❤️ 
