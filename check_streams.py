from pylsl import resolve_stream

def main():
    print("Procurando streams LSL...")
    streams = resolve_stream()
    if not streams:
        print("Nenhum stream encontrado.")
    else:
        for s in streams:
            print(f"Stream encontrado: {s.name()} - Tipo: {s.type()}")

if __name__ == "__main__":
    main()
