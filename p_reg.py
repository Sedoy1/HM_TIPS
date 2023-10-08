import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np


def p(state, params):
    """
    расчет управляющего воздействия на основе П-регулятора
    :param state: состояния ОУ
    :param params: параметры П-регулятора
    :return: управляющее воздействие
    """

    # Коэффициент П-регулятора
    kp_alt = params[0]  # пропорциональная составляющая по x
    kp_ang = params[1]  # пропорциональная составляющая по углу

    # расчет целевой переменной
    alt_tgt = np.abs(state[0])
    ang_tgt = (.25 * np.pi) * (state[0] + state[2])

    # расчет ошибки
    alt_error = (alt_tgt - state[1])
    ang_error = (ang_tgt - state[4])

    # Формируем управляющее воздействие П-регулятора
    alt_adj = kp_alt * alt_error
    ang_adj = kp_ang * ang_error

    # Приводим к интервалу (-1,  1)
    a = np.array([alt_adj, ang_adj])
    a = np.clip(a, -1, +1)

    # Если есть точка соприкосновения с землей, то глушим двигатели, никакие действия не передаем
    if state[6] or state[7]:
        a[:] = 0
    return a


def start_game(environment, params, video_recorder=False):
    """
    Симуляция
    :param environment: среда Gym
    :param params: параметры П-регулятора
    :param video_recorder: объект для записи видео. False - без записи видео
    :return: суммарное качество посадки
    """
    state, _ = environment.reset()
    done = False
    total = 0
    while not done:
        environment.render()
        if video_recorder:
            video_recorder.capture_frame()

        # П-регулятор
        action = p(state, params)
        state, reward, done, info, _ = environment.step(action)
        total += reward

    return total


if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    env = gym.make(env_name,
                   render_mode="rgb_array",
                   continuous=True)

    print('Размер вектора состояния ОУ: ', env.observation_space.shape)
    print('Структура управляющего воздействия', env.action_space)
    params_pd = np.array([0.85767974, -0.85962648, 4.53547841, 0.70256431])

    vid = VideoRecorder(env, path=f"random_luna_lander_p_reg.mp4")
    score = start_game(env, params_pd, video_recorder=vid)

    vid.close()

    env.close()
