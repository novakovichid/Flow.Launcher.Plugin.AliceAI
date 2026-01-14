# Flow Launcher LLM Plugin (OpenAI + Yandex)
Этот плагин позволяет использовать модели OpenAI и Яндекс (YandexGPT) в [Flow Launcher](https://www.flowlauncher.com/).

![Demo video of the Flow Launcher AliceAI Plugin](https://i.imgur.com/WQwNY7y.gif)

## Возможности
- 🔌 Поддержка OpenAI и Yandex Cloud Foundation Models
- 🔁 Выбор способа подключения Яндекса: нативный API или OpenAI-совместимый endpoint
- 🧠 Быстрый выбор модели из списка
- 📝 Ключевые слова для коротких, длинных или стандартных ответов
- 💬 Пользовательские системные подсказки через CSV
- 🗃️ Копирование ответа или открытие в текстовом файле
- ✋ Запуск запроса по стоп-ключу

## Требования
### OpenAI
1. Учётная запись OpenAI.
2. Подключённая оплата в профиле OpenAI: https://platform.openai.com/account/billing/payment-methods
3. API-ключ OpenAI: https://platform.openai.com/account/api-keys

### Yandex Cloud
1. Проект в Yandex Cloud.
2. Folder ID (ID каталога).
3. Один из способов авторизации:
   - API Key (Authorization: Api-Key ...)
   - IAM Token (Authorization: Bearer ...)

## Установка
1. Установите [Flow Launcher](https://www.flowlauncher.com/).
2. Откройте Flow Launcher и введите `Settings`.
3. Перейдите в модуль `Plugin Store`.
4. Найдите `AliceAI`.
5. Нажмите `Install`.
6. Flow Launcher перезапустится автоматически (или сделайте это вручную).
7. Откройте `Plugins` -> `AliceAI`.
8. Заполните нужные настройки (см. ниже).
9. Выполните команду `Save Settings` в Flow Launcher.

## Использование
### Базовый сценарий
1. Введите ключевое слово `ai`.
2. Наберите запрос и добавьте стоп-ключ в конце (по умолчанию `||`).
3. Дождитесь обновления списка.
4. Скопируйте ответ или откройте его в текстовом файле.

### Системные подсказки
Системные подсказки определяют стиль ответа. Они выбираются по ключевому слову в начале запроса. Если ключевое слово не найдено, используется значение `Default system prompt`.

По умолчанию доступны:
|Ключевое слово|Системная подсказка|
|---|---|
|normal|You are an all-knowing AI bot.|
|short|You are an all-knowing AI bot. All your answers are short, to the point, and don't give any additional context.|
|long|You are an all-knowing AI bot. All your answers are in-depth and give both a step-by-step explanation how you came to that answer, as well as references to the resources you used.|

## Добавление своих системных подсказок
1. Откройте Flow Launcher.
2. Введите `Settings`.
3. Перейдите в `Plugins -> AliceAI`.
4. Нажмите иконку папки.
5. Откройте `system_messages.csv`.
6. В первом столбце укажите ключевое слово (без пробелов).
7. Во втором столбце укажите системную подсказку.
8. Сохраните файл.

Подборка готовых подсказок: https://github.com/f/awesome-chatgpt-prompts

## Настройки
|Настройка|Описание|Значение по умолчанию|
|---|---|---|
|Action keyword|Ключевое слово запуска плагина|`ai`|
|Provider|Провайдер LLM: `openai`, `yandex_native`, `yandex_openai`|`openai`|
|OpenAI API Key|Ключ OpenAI|`(пусто)`|
|Model|Модель OpenAI|`gpt-5-mini`|
|OpenAI request mode|Тип запроса OpenAI: `sync` или `async`|`sync`|
|API Endpoint|Endpoint OpenAI (или OpenAI-совместимый)|`https://api.openai.com/v1/chat/completions`|
|Yandex auth type|Тип авторизации: `api_key` или `iam_token`|`api_key`|
|Yandex API Key|API Key из Yandex Cloud|`(пусто)`|
|Yandex IAM Token|IAM токен из Yandex Cloud|`(пусто)`|
|Yandex Folder ID|ID каталога в Yandex Cloud|`(пусто)`|
|Yandex model (preset)|Готовые варианты моделей Яндекса (dropdown)|`yandexgpt/latest`|
|Yandex model (legacy/manual)|Модель Яндекса: идентификатор (`yandexgpt/latest`) или полный URI (`gpt://<folder-id>/<model>`), используется если preset пустой|`yandexgpt/latest`|
|Yandex model (custom)|Кастомная модель/URI, используется если preset = `custom`|`(пусто)`|
|Yandex request mode|Тип запроса Яндекса: `sync` или `async`|`sync`|
|Yandex native endpoint|Нативный endpoint Foundation Models API|`https://llm.api.cloud.yandex.net/foundationModels/v1/completion`|
|Yandex OpenAI-compatible endpoint|OpenAI-совместимый endpoint Яндекса|`https://llm.api.cloud.yandex.net/v1/chat/completions`|
|Prompt stop|Стоп-символы запроса|`||`|
|Default system prompt|Ключ по умолчанию для системной подсказки|`normal`|
|Save conversation|Сохранять историю запросов|`false`|
|Log Level|Уровень логирования|`error`|

## Примечания по моделям
- OpenAI: добавлены актуальные модели API (семейство GPT-5/5.2, GPT-4o и GPT-4.1). При необходимости используйте кастомный endpoint.
- Yandex native: принимает полный `modelUri` вида `gpt://<folder-id>/<model>` или короткий идентификатор модели (тогда нужен Folder ID).
- Yandex OpenAI-compatible: отправляет модель как `gpt://<folder-id>/<model>` (если Folder ID заполнен), либо принимает полный URI напрямую.
- Приоритет выбора модели: preset (если не `custom`) → custom (если задан) → legacy/manual.

### Актуальные модели Yandex (идентификаторы)
Можно указывать короткий идентификатор (с Folder ID) или полный URI `gpt://<folder-id>/<model>`.

Примеры идентификаторов:
- `aliceai-llm`
- `yandexgpt/rc` (YandexGPT Pro 5.1)
- `yandexgpt/latest` (YandexGPT Pro 5)
- `yandexgpt-lite`
- `qwen3-235b-a22b-fp8/latest`
- `gpt-oss-120b/latest`
- `gpt-oss-20b/latest`
- `gemma-3-27b-it/latest`

# Backlog
* Ability to take into account the context of the previous prompts.
