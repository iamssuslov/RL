# Reinforcement Learning для задачи CartPole-v1

Этот проект реализует алгоритмы обучения с подкреплением для решения задачи `CartPole-v1` из среды OpenAI Gym.

## Описание задачи

**CartPole-v1** — классическая задача балансировки шеста (pole), закрепленного на тележке (cart).  
**Цель**: удерживать шест в вертикальном положении как можно дольше, управляя тележкой.  

- **Действия**:  
  - `0` — толкать тележку влево  
  - `1` — толкать тележку вправо  
- **Наблюдение (состояние)**:  
  - Позиция тележки (`x`)  
  - Скорость тележки (`v`)  
  - Угол наклона шеста (`θ`)  
  - Угловая скорость шеста (`ω`)  
- **Награда**: `+1` за каждый шаг, пока шест не упал.  
- **Эпизод завершается**, если:  
  - Шест отклоняется от вертикали более чем на 15°.  
  - Тележка смещается от центра более чем на 2.4 единицы.  
  - Длина эпизода превышает 500 шагов (максимум для `v1`).  

## Установка и запуск

1. Установите зависимости:
   ```bash
   pip install gym numpy matplotlib stable-baselines3

2. Запустите файл скрипта:
   ```bash
   python main.py