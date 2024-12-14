from signals.eeg import process_eeg, calculate_band_powers
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream

def main():
    print("Procurando streams LSL...")
    streams = resolve_stream()
    print(f"Streams encontrados: {[s.name() for s in streams]}")

    # Conectar ao stream EEG
    try:
        eeg_inlet = StreamInlet([s for s in streams if s.type() == 'EEG'][0])
        print("Conectado ao stream EEG.")
    except IndexError as e:
        print(f"Erro ao conectar aos streams: {e}")
        return

    # Buffers para sinais
    eeg_buffer = np.zeros((5, 250))  # 5 canais EEG

    # Configurar gráficos
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Gráfico de EEG
    axs[0].set_ylim(-100, 100)  # Ajuste de escala para sinais filtrados
    eeg_lines = axs[0].plot(np.zeros((5, 250)).T)

    # Gráfico de potências das bandas
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    axs[1].set_ylim(0, 1000)  # Ajuste inicial da escala para valores atualizados
    band_bars = axs[1].bar(bands, [0] * len(bands))


    while True:
        try:
            # Processar o sinal EEG
            eeg_buffer = process_eeg(eeg_inlet, eeg_buffer)
            band_powers = calculate_band_powers(eeg_buffer, fs=256)

            # Atualizar gráfico de EEG
            for i, line in enumerate(eeg_lines):
                line.set_ydata(eeg_buffer[i, -250:])  # Últimas 250 amostras

            # Atualizar gráfico de potências das bandas
            band_values = [np.mean(band_powers[band]) for band in bands]
            for bar, value in zip(band_bars, band_values):
                bar.set_height(value)

            plt.pause(0.01)

        except KeyboardInterrupt:
            print("Encerrando...")
            plt.close()
            break


if __name__ == "__main__":
    main()
