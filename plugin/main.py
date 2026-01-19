# -*- coding: utf-8 -*-

import os
import csv
import logging
import time
from datetime import datetime
from flox import Flox  # noqa: E402
import webbrowser  # noqa: E402
import requests  # noqa: E402
import json  # noqa: E402
import pyperclip  # noqa: E402
from typing import Tuple, Optional

PROXIES = {
    "http": os.environ.get("HTTP_PROXY", ""),
    "https": os.environ.get("HTTPS_PROXY", ""),
}


class AliceAI(Flox):
    def __init__(self):
        self._load_settings()

        try:
            self.csv_file = open("system_messages.csv", encoding="utf-8", mode="r")
            reader = csv.DictReader(self.csv_file, delimiter=";")
            self.prompts = list(reader)
            [logging.debug(f"Found prompt: {row}") for row in self.prompts]

        except FileNotFoundError:
            self.prompts = None
            logging.error("Unable to open system_messages.csv")

    def _load_settings(self) -> None:
        self.provider = (self.settings.get("provider") or "openai").lower()
        self.api_key = self.settings.get("api_key")
        model_setting = self.settings.get("model") or "gpt-5-mini"
        self.model = self._normalize_model_option(model_setting)
        self.prompt_stop = self.settings.get("prompt_stop")
        self.default_system_prompt = self.settings.get("default_prompt")
        self.custom_system_prompt = self.settings.get("custom_system_prompt") or ""
        self.save_conversation_setting = self.settings.get("save_conversation")
        self.request_history_limit = self._parse_int_setting(
            self.settings.get("request_history_limit"), 10
        )
        self.log_level = self.settings.get("log_level")
        self.api_endpoint = (
            self.settings.get("api_endpoint")
            or "https://api.openai.com/v1/chat/completions"
        )
        self.openai_request_mode = (
            self.settings.get("openai_request_mode") or "sync"
        ).lower()
        self.yandex_request_mode = (
            self.settings.get("yandex_request_mode") or "sync"
        ).lower()
        self.yandex_auth_type = (
            self.settings.get("yandex_auth_type") or "api_key"
        ).lower()
        self.yandex_api_key = self.settings.get("yandex_api_key")
        self.yandex_iam_token = self.settings.get("yandex_iam_token")
        self.yandex_folder_id = self.settings.get("yandex_folder_id")
        self.yandex_model = self.settings.get("yandex_model") or "yandexgpt/latest"
        yandex_model_preset_setting = self.settings.get("yandex_model_preset") or ""
        self.yandex_model_preset = self._normalize_model_option(
            yandex_model_preset_setting
        )
        self.yandex_model_custom = self.settings.get("yandex_model_custom") or ""
        self.yandex_native_endpoint = (
            self.settings.get("yandex_native_endpoint")
            or "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        )
        self.yandex_openai_endpoint = (
            self.settings.get("yandex_openai_endpoint")
            or "https://llm.api.cloud.yandex.net/v1/chat/completions"
        )
        self.answer_action_order = self.settings.get("answer_action_order") or (
            "copy,preview,editor"
        )
        self.enable_copy_action = self._parse_bool_setting(
            self.settings.get("enable_copy_action"), True
        )
        self.enable_preview_action = self._parse_bool_setting(
            self.settings.get("enable_preview_action"), True
        )
        self.enable_editor_action = self._parse_bool_setting(
            self.settings.get("enable_editor_action"), True
        )
        self.copy_action_mode = (
            self.settings.get("copy_action_mode") or "answer_only"
        ).lower()
        self.preview_action_mode = (
            self.settings.get("preview_action_mode") or "answer_only"
        ).lower()
        self.editor_action_mode = (
            self.settings.get("editor_action_mode") or "answer_only"
        ).lower()
        self.editor_open_mode = (
            self.settings.get("editor_open_mode") or "saved_if_available"
        ).lower()
        self.logger_level(self.log_level)

    def query(self, query: str) -> None:
        self._load_settings()
        if not self._ensure_auth():
            return
        if self.prompts is None:
            self.add_item(
                title="Unable to load the system prompts from CSV",
                subtitle="Please validate that the plugins folder contains a valid system_prompts.csv",  # noqa: E501
                method=self.open_plugin_folder,
            )
            return
        if query.endswith(self.prompt_stop):
            prompt, prompt_keyword, system_message = self.split_prompt(query)

            answer, prompt_timestamp, answer_timestamp = self.send_prompt(
                prompt, system_message
            )

            self._log_request_history(
                prompt_keyword,
                prompt,
                system_message,
                answer,
                prompt_timestamp,
                answer_timestamp,
            )

            filename = None
            if self.save_conversation_setting:
                filename = self.save_conversation(
                    prompt_keyword, prompt, prompt_timestamp, answer, answer_timestamp
                )

            if answer:
                answer = answer.lstrip("\n").lstrip("\n")
                short_answer = self.ellipsis(answer, 30)

                for action in self._build_answer_actions(
                    prompt, answer, filename, short_answer
                ):
                    self.add_item(**action)

        else:
            self.add_item(
                title=f"Type your prompt and end with {self.prompt_stop}",
                subtitle=(
                    f"Current model: {self._current_model_label()} "
                    f"| Mode: {self._current_request_mode_label()}"
                ),
            )
        return

    def send_prompt(
        self, prompt: str, system_message: str
    ) -> Tuple[str, datetime, datetime]:
        """
        Query the selected provider end-point
        """
        if self.provider == "openai":
            return self._send_openai_prompt(prompt, system_message)
        if self.provider == "yandex_openai":
            return self._send_yandex_openai_prompt(prompt, system_message)
        return self._send_yandex_native_prompt(prompt, system_message)

    def _send_openai_prompt(
        self, prompt: str, system_message: str
    ) -> Tuple[str, datetime, datetime]:
        url = self.api_endpoint

        headers = self._openai_headers()
        stream = self.openai_request_mode == "async"
        body = self._openai_body(prompt, system_message, self.model, stream)

        return self._send_request(url, headers, body, "OpenAI", stream)

    def _send_yandex_openai_prompt(
        self, prompt: str, system_message: str
    ) -> Tuple[str, datetime, datetime]:
        url = self.yandex_openai_endpoint
        headers = self._yandex_headers()
        stream = self.yandex_request_mode == "async"
        body = self._openai_body(
            prompt, system_message, self._yandex_model_value(), stream
        )

        return self._send_request(
            url, headers, body, "Yandex OpenAI-compatible", stream
        )

    def _send_yandex_native_prompt(
        self, prompt: str, system_message: str
    ) -> Tuple[str, datetime, datetime]:
        if self.yandex_request_mode == "async":
            return self._send_yandex_native_async_prompt(prompt, system_message)
        url = self.yandex_native_endpoint
        headers = self._yandex_headers()
        model_uri = self._yandex_model_uri()
        body = {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": 2000,
            },
            "messages": [
                {"role": "system", "text": system_message},
                {"role": "user", "text": prompt},
            ],
        }

        prompt_timestamp = datetime.now()
        logging.debug(f"Sending Yandex native request with data: {body}")
        try:
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=json.dumps(body),
                proxies=PROXIES,
            )
        except UnicodeEncodeError as e:
            logging.error(f"UnicodeEncodeError: {e}")
            return "", prompt_timestamp, datetime.now()

        logging.debug(f"Response: {response}")
        answer_timestamp = datetime.now()
        result = ""
        response_json = response.json()
        if response.ok:
            alternatives = response_json.get("result", {}).get("alternatives", [])
            for entry in alternatives:
                message = entry.get("message", {})
                result += message.get("text", "")
        else:
            self._handle_error(response, response_json, "Yandex native")

        return result, prompt_timestamp, answer_timestamp

    def _send_yandex_native_async_prompt(
        self, prompt: str, system_message: str
    ) -> Tuple[str, datetime, datetime]:
        url = self._yandex_async_endpoint(self.yandex_native_endpoint)
        headers = self._yandex_headers()
        model_uri = self._yandex_model_uri()
        body = {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": 2000,
            },
            "messages": [
                {"role": "system", "text": system_message},
                {"role": "user", "text": prompt},
            ],
        }

        prompt_timestamp = datetime.now()
        logging.debug(f"Sending Yandex native async request with data: {body}")
        try:
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=json.dumps(body),
                proxies=PROXIES,
            )
        except UnicodeEncodeError as e:
            logging.error(f"UnicodeEncodeError: {e}")
            return "", prompt_timestamp, datetime.now()

        logging.debug(f"Response: {response}")
        answer_timestamp = datetime.now()
        if not response.ok:
            response_json = response.json()
            self._handle_error(response, response_json, "Yandex native async")
            return "", prompt_timestamp, answer_timestamp

        response_json = response.json()
        operation_id = response_json.get("id")
        if not operation_id:
            logging.error("Missing operation id in Yandex async response.")
            return "", prompt_timestamp, answer_timestamp

        return self._poll_yandex_operation(
            operation_id, headers, prompt_timestamp, answer_timestamp
        )

    def _poll_yandex_operation(
        self,
        operation_id: str,
        headers: dict,
        prompt_timestamp: datetime,
        answer_timestamp: datetime,
        max_attempts: int = 60,
        poll_interval_seconds: float = 1.0,
    ) -> Tuple[str, datetime, datetime]:
        operation_url = self._yandex_operation_endpoint(operation_id)
        for _ in range(max_attempts):
            try:
                response = requests.get(
                    operation_url, headers=headers, proxies=PROXIES
                )
            except UnicodeEncodeError as e:
                logging.error(f"UnicodeEncodeError: {e}")
                return "", prompt_timestamp, datetime.now()

            if not response.ok:
                response_json = response.json()
                self._handle_error(response, response_json, "Yandex native async")
                return "", prompt_timestamp, datetime.now()

            response_json = response.json()
            if response_json.get("done"):
                if response_json.get("error"):
                    self._handle_error(
                        response, response_json, "Yandex native async"
                    )
                    return "", prompt_timestamp, datetime.now()
                answer_timestamp = datetime.now()
                response_body = response_json.get("response", {})
                result = ""
                alternatives = response_body.get("alternatives")
                if alternatives is None:
                    alternatives = response_body.get("result", {}).get(
                        "alternatives", []
                    )
                for entry in alternatives:
                    message = entry.get("message", {})
                    result += message.get("text", "")
                return result, prompt_timestamp, answer_timestamp

            time.sleep(poll_interval_seconds)

        logging.error("Timed out waiting for Yandex async operation to complete.")
        return "", prompt_timestamp, datetime.now()

    def _yandex_async_endpoint(self, endpoint: str) -> str:
        if endpoint.endswith("completionAsync"):
            return endpoint
        if endpoint.endswith("/completion"):
            return f"{endpoint}Async"
        return "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"

    def _yandex_operation_endpoint(self, operation_id: str) -> str:
        return f"https://operation.api.cloud.yandex.net/operations/{operation_id}"

    def _send_request(
        self,
        url: str,
        headers: dict,
        body: dict,
        provider_label: str,
        stream: bool = False,
    ) -> Tuple[str, datetime, datetime]:
        data = json.dumps(body)
        prompt_timestamp = datetime.now()
        logging.debug(f"Sending request with data: {data}")
        try:
            response = requests.request(
                "POST", url, headers=headers, data=data, proxies=PROXIES, stream=stream
            )
        except UnicodeEncodeError as e:
            logging.error(f"UnicodeEncodeError: {e}")
            return "", prompt_timestamp, datetime.now()

        logging.debug(f"Response: {response}")
        answer_timestamp = datetime.now()

        result = ""
        if stream:
            result = self._consume_openai_stream(response)
        else:
            response_json = response.json()
            if response.ok:
                for entry in response_json.get("choices", []):
                    message = entry.get("message", {})
                    result += message.get("content", "")
            else:
                self._handle_error(response, response_json, provider_label)
        return result, prompt_timestamp, answer_timestamp

    def _consume_openai_stream(self, response) -> str:
        if not response.ok:
            response_json = response.json()
            self._handle_error(response, response_json, "Streaming request")
            return ""
        result = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            line = line.strip()
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if not line or line == "[DONE]":
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for entry in payload.get("choices", []):
                delta = entry.get("delta", {})
                if delta:
                    result += delta.get("content", "")
                else:
                    message = entry.get("message", {})
                    result += message.get("content", "")
        return result

    def _consume_yandex_stream(self, response) -> str:
        if not response.ok:
            response_json = response.json()
            self._handle_error(response, response_json, "Yandex native streaming")
            return ""
        result = ""
        last_message_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            line = line.strip()
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            alternatives = payload.get("result", {}).get("alternatives", [])
            for entry in alternatives:
                message = entry.get("message", {})
                message_text = message.get("text", "")
                if not message_text:
                    continue
                if message_text.startswith(last_message_text):
                    result += message_text[len(last_message_text) :]
                else:
                    result += message_text
                last_message_text = message_text
        return result

    def _handle_error(self, response, response_json: dict, provider_label: str) -> None:
        error_message = (
            response_json.get("error", {}).get("message")
            or response_json.get("message")
            or "Unknown error"
        )
        self.add_item(title="An error occurred", subtitle=error_message)
        logging.error(
            f"{provider_label} API returned {response.status_code} with message: {response_json}"
        )

    def _openai_headers(self) -> dict:
        return {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

    def _yandex_headers(self) -> dict:
        token = self._yandex_token()
        headers = {
            "Authorization": f"{self._yandex_auth_prefix()} {token}",
            "Content-Type": "application/json",
        }
        if self.yandex_folder_id:
            headers["x-folder-id"] = self.yandex_folder_id
        return headers

    def _openai_body(
        self, prompt: str, system_message: str, model: str, stream: bool
    ) -> dict:
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
        }

    def _yandex_auth_prefix(self) -> str:
        return "Api-Key" if self.yandex_auth_type == "api_key" else "Bearer"

    def _yandex_token(self) -> str:
        if self.yandex_auth_type == "iam_token":
            return self.yandex_iam_token or ""
        return self.yandex_api_key or ""

    def _yandex_model_uri(self) -> str:
        model = self._yandex_model_raw()
        if model.startswith("gpt://"):
            return model
        if self.yandex_folder_id:
            return f"gpt://{self.yandex_folder_id}/{model}"
        return ""

    def _yandex_model_raw(self) -> str:
        preset = (self.yandex_model_preset or "").strip()
        custom = (self.yandex_model_custom or "").strip()
        if preset and preset != "custom":
            return preset
        if custom:
            return custom
        return (self.yandex_model or "").strip()

    def _yandex_model_value(self) -> str:
        return self._yandex_model_uri() or self._yandex_model_raw()

    def _current_model_label(self) -> str:
        if self.provider == "openai":
            return self.model
        model_label = self._yandex_model_raw() or self.yandex_model
        if self.provider == "yandex_openai":
            return f"{model_label} (OpenAI-compatible)"
        return f"{model_label} (native)"

    def _current_request_mode_label(self) -> str:
        if self.provider == "openai":
            mode = self.openai_request_mode
        else:
            mode = self.yandex_request_mode
        if mode == "sync":
            return "sync (blocking)"
        if self.provider == "yandex_native":
            return "async (operation)"
        return "async (streaming)"

    def _ensure_auth(self) -> bool:
        if self.provider == "openai":
            if not self.api_key:
                self.add_item(
                    title="Unable to load the OpenAI API key",
                    subtitle=(
                        "Please make sure you've added a valid OpenAI API key in the settings"
                    ),
                )
                return False
            return True

        token = self._yandex_token()
        if not token:
            self.add_item(
                title="Unable to load the Yandex token",
                subtitle=(
                    "Please make sure you've added a valid Yandex API key or IAM token"
                ),
            )
            return False
        if not self._yandex_model_value():
            self.add_item(
                title="Missing Yandex model",
                subtitle="Please select or enter a Yandex model in the settings",
            )
            return False
        if (
            self.provider in ("yandex_native", "yandex_openai")
            and not self._yandex_model_uri()
        ):
            self.add_item(
                title="Missing Yandex Folder ID",
                subtitle=(
                    "Please provide a folder ID or use a full model URI "
                    "(gpt://<folder-id>/<model>)"
                ),
            )
            return False
        return True

    def save_conversation(
        self,
        keyword: str,
        prompt: str,
        prompt_timestamp: datetime,
        answer: str,
        answer_timestamp: datetime,
    ) -> str:
        filename = f"Conversations '{keyword}' keyword.txt"
        formatted_prompt_timestamp = prompt_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        formatted_answer_timestamp = answer_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        new_content = f"[{formatted_prompt_timestamp}] User: {prompt}\n[{formatted_answer_timestamp}] AliceAI: {answer}\n\n"  # noqa: E501

        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as file:
                    existing_content = file.read()
            except PermissionError:
                logging.error(PermissionError)
        else:
            existing_content = ""

        new_content = new_content + existing_content

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(new_content)
        except PermissionError:
            logging.error(PermissionError)

        return filename

    def split_prompt(self, query: str) -> Tuple[str, str, str]:
        prompt = query.rstrip(self.prompt_stop).strip()
        prompt_array = prompt.split(" ")
        prompt_keyword = prompt_array[0].lower()

        system_message = ""

        for row in self.prompts:
            if row["Key Word"] == prompt_keyword:
                system_message = row["System Message"]
                prompt = prompt.split(" ", 1)[1]

        if not system_message:
            prompt_keyword = self.default_system_prompt

            for row in self.prompts:
                if row["Key Word"] == self.default_system_prompt:
                    system_message = row["System Message"]

        if len(prompt_array) == 1:
            prompt = prompt_array[0]

        system_message = self._apply_custom_system_prompt(system_message)

        logging.debug(
            f"""
        Prompt: {prompt}
        Prompt keyword: {prompt_keyword}
        System message: {system_message}
        """
        )

        return prompt, prompt_keyword, system_message

    def _apply_custom_system_prompt(self, system_message: str) -> str:
        custom_prompt = self.custom_system_prompt.strip()
        if not custom_prompt:
            return system_message
        if not system_message:
            return custom_prompt
        return f"{system_message}\n\n{custom_prompt}"

    def _parse_int_setting(self, value, fallback: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if parsed > 0 else fallback

    def _parse_bool_setting(self, value, fallback: bool) -> bool:
        if value is None:
            return fallback
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes", "on")

    def _normalize_model_option(self, value: str) -> str:
        if not value:
            return value
        if " — " in value:
            return value.split(" — ", 1)[0].strip()
        return value.strip()

    def _log_request_history(
        self,
        prompt_keyword: str,
        prompt: str,
        system_message: str,
        answer: str,
        prompt_timestamp: datetime,
        answer_timestamp: datetime,
    ) -> None:
        if self.request_history_limit <= 0:
            return
        history_file = os.path.join(os.getcwd(), "request_history.json")
        entry = {
            "prompt_keyword": prompt_keyword,
            "prompt": prompt,
            "system_message": system_message,
            "answer": answer,
            "provider": self.provider,
            "model": self._current_model_label(),
            "prompt_timestamp": prompt_timestamp.isoformat(),
            "answer_timestamp": answer_timestamp.isoformat(),
        }
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as file:
                    history = json.load(file)
            except (json.JSONDecodeError, OSError):
                history = []
        if not isinstance(history, list):
            history = []
        history.insert(0, entry)
        history = history[: self.request_history_limit]
        try:
            with open(history_file, "w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)
        except OSError as error:
            logging.error(f"Failed to write request history: {error}")

    def ellipsis(self, string: str, length: int):
        string = string.split("\n", 1)[0]
        return string[: length - 3] + "..." if len(string) > length else string

    def _build_answer_actions(
        self, prompt: str, answer: str, filename: Optional[str], short_answer: str
    ) -> list:
        action_order = self._parse_action_order(self.answer_action_order)
        action_text = {
            "copy": self._format_action_text(prompt, answer, self.copy_action_mode),
            "preview": self._format_action_text(
                prompt, answer, self.preview_action_mode
            ),
            "editor": self._format_action_text(
                prompt, answer, self.editor_action_mode
            ),
        }
        action_definitions = {
            "copy": {
                "title": "Copy to clipboard",
                "subtitle": f"Answer: {short_answer}",
                "method": self.copy_answer,
                "parameters": [action_text["copy"]],
                "enabled": self.enable_copy_action,
            },
            "preview": {
                "title": "Preview answer",
                "subtitle": f"Answer: {short_answer}",
                "method": self.display_answer,
                "parameters": [action_text["preview"]],
                "enabled": self.enable_preview_action,
                "dont_hide": True,
                "Preview": {"Description": action_text["preview"]},
            },
            "editor": {
                "title": "Open in text editor",
                "subtitle": f"Answer: {short_answer}",
                "method": self.open_in_editor,
                "parameters": [
                    filename,
                    action_text["editor"],
                    self.editor_open_mode,
                ],
                "enabled": self.enable_editor_action,
            },
        }
        actions = []
        for action_id in action_order:
            action = action_definitions.get(action_id)
            if not action or not action["enabled"]:
                continue
            if not action_text.get(action_id):
                continue
            action_payload = dict(action)
            action_payload.pop("enabled", None)
            actions.append(action_payload)
        return actions

    def _parse_action_order(self, value: str) -> list:
        known = ["copy", "preview", "editor"]
        tokens = [token.strip().lower() for token in (value or "").split(",")]
        order = []
        seen = set()
        for token in tokens:
            if token and token in known and token not in seen:
                order.append(token)
                seen.add(token)
        for token in known:
            if token not in seen:
                order.append(token)
        return order

    def _format_action_text(
        self, prompt: str, answer: str, mode: str
    ) -> str:
        if not answer:
            return ""
        if mode == "prompt_and_answer" and prompt:
            return f"Prompt:\n{prompt}\n\nAnswer:\n{answer}"
        return answer

    def copy_answer(self, text: str) -> None:
        """
        Copy answer to the clipboard.
        """
        if not text:
            return
        pyperclip.copy(text)

    def open_in_editor(
        self,
        filename: Optional[str],
        text: Optional[str],
        open_mode: str = "saved_if_available",
    ) -> None:
        """
        Open the answer in the default text editor. If no filename is given,
        the conversation will be written to a new text file and opened.
        """
        if filename and open_mode == "saved_if_available":
            webbrowser.open(filename)
            return

        if text:
            temp_file = "temp_answer.txt"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(text)
            webbrowser.open(temp_file)
            return

    def display_answer(self, text: str) -> None:
        """
        Display the answer in Flow Launcher preview panel.
        """
        if not text:
            return
        self.add_item(
            title="Answer preview",
            subtitle="Preview panel",
            dont_hide=True,
            Preview={"Description": text},
        )
        return

    def open_plugin_folder(self) -> None:
        webbrowser.open(os.getcwd())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.csv_file.close()


if __name__ == "__main__":
    AliceAI()
