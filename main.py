from signals.eeg import process_eeg
from signals.ppg import process_ppg
from signals.respiration import process_respiration
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream

def main():
    print("Procurando streams LSL...")
    streams = resolve_stream()
    print(f"Streams encontrados: {[s.name() for s in streams]}")

    # Filtrar streams de EEG, PPG e ACC
    try:
        eeg_inlet = StreamInlet([s for s in streams if s.type() == 'EEG'][0])
        print("Conectado ao stream EEG.")
        ppg_inlet = StreamInlet([s for s in streams if s.type() == 'PPG'][0])
        print("Conectado ao stream PPG.")
        acc_inlet = StreamInlet([s for s in streams if s.type() == 'ACC'][0])
        print("Conectado ao stream ACC.")
    except IndexError as e:
        print(f"Erro ao conectar aos streams: {e}")
        return

    print("Conexão estabelecida com todos os streams!")

    # Buffers para sinais
    eeg_buffer = np.zeros((5, 250))  # 5 canais EEG
    ppg_buffer = np.zeros((1, 250))  # 1 canal PPG
    resp_buffer = np.zeros((3, 250))  # 3 canais ACC

    # Configurar gráficos
    print("Configurando gráficos...")
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[0].set_ylim(-100, 100)  # Escala para EEG após filtragem
    axs[1].set_ylim(0, 200)     # Escala para Batimentos Cardíacos
    axs[2].set_ylim(-0.1, 0.1)  # Escala para Respiração

    eeg_lines = axs[0].plot(np.zeros((5, 250)).T)
    ppg_line, = axs[1].plot(np.zeros(250))
    resp_line, = axs[2].plot(np.zeros(250))

    print("Gráficos configurados. Iniciando processamento...")
    try:
        while True:
            print("Processando sinais...")
            # Processar cada tipo de sinal
            eeg_buffer = process_eeg(eeg_inlet, eeg_buffer)
            print("EEG processado.")
            ppg_buffer, bpm = process_ppg(ppg_inlet, ppg_buffer)
            print(f"PPG processado. BPM: {bpm:.2f}")
            resp_buffer, resp_signal = process_respiration(acc_inlet, resp_buffer)
            print("Respiração processada.")

            # Garantir que os dados plotados correspondem ao tamanho esperado
            for i, line in enumerate(eeg_lines):
                line.set_ydata(eeg_buffer[i, -250:])  # Últimas 250 amostras
            ppg_line.set_ydata(ppg_buffer[0, -250:])  # Últimas 250 amostras do PPG
            resp_line.set_ydata(resp_signal[-250:])  # Últimas 250 amostras de respiração

            # Atualizar título com BPM
            axs[1].set_title(f"Batimentos Cardíacos (BPM): {bpm:.2f}")
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Encerrando...")
        plt.close()


if __name__ == "__main__":
    main()
