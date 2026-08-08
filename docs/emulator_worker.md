# Worker de emulador headless "on-demand" para desambiguação em tempo real

> Status: **plano investigado e aprovado, execução ainda não iniciada** (2026-08-07). Este documento
> registra o contexto e a evidência real levantada antes de qualquer código - próxima sessão deve
> começar por M1, não redescobrir o que já está aqui.

## Contexto

O objetivo é fechar o loop entre gameplay real e o pipeline `draft` (ver `draft_graph.py`): um
humano joga com `PORTAL_TRACE=1` ligado, gerando um `portal_trace.jsonl` como um "vídeo" de eventos
(isso **já funciona** hoje, é o mesmo mecanismo usado no case study de consolidação do `draft`). O
que falta é a peça nova: o agente/LLM poder, no **seu próprio tempo** (não em tempo real de 60fps),
pedir para um emulador headless avançar frames a partir de um savestate e devolver dados novos
(trace, memória) - para desambiguar um `JMP @Rn`/`BRAF Rn` sem alvo estático resolvido, por exemplo,
sem depender do humano estar jogando exatamente aquele trecho na hora.

## Investigação real (evidência por arquivo:linha, projeto de origem
`jhonatanTeixeira/portal_to_another_world`)

**Existe de verdade:**
- `retro_serialize_size`/`retro_serialize`/`retro_unserialize` (savestate) e `retro_run()` (avança
  exatamente 1 frame, API libretro padrão, sem parâmetros) em
  `vendor/yabassanshiro/yabause/src/libretro/libretro.c:938-977,1473-1509`.
- `retro_get_memory_data`/`retro_get_memory_size` (`libretro.c:1435,1440`) - leitura de memória do
  core já exposta pela API padrão, sem precisar inventar nada novo.
- `portal_trace.c/.h` - já grava JSONL (`call`/`return`/`mem_write`/`frame`) durante qualquer sessão
  real, compilado com `PORTAL_TRACE=1`. É só um logger, não um mecanismo de replay de input.

**NÃO existe (confirmado, precisa ser construído):**
- Nenhum suporte a "movie"/BSV/replay determinístico de input no core - nem no yabassanshiro, nem
  vendorizado do RetroArch nesse repositório.
- Nenhum socket/pipe/IPC de controle externo no core - `portal_trace` só tem uma env var de path e
  uma hotkey de teclado (F9), nada que um processo externo possa chamar para "avançar N frames
  agora".
- Nenhum front-end headless já implementado (nem C, nem Python) fora do `main.rs` do projeto Saturn
  (que é seu próprio mini-driver Rust, não usa o core C real).

**Risco real confirmado (não hipotético):** o core **negocia contexto OpenGL obrigatoriamente**
(`RETRO_ENVIRONMENT_SET_HW_RENDER`, pedindo `RETRO_HW_CONTEXT_OPENGLES3` depois `OPENGL` como
fallback - `libretro.c:695-705`), com callbacks `context_reset`/`context_destroy`/
`get_current_framebuffer` (`libretro.c:553,659,682`). Um front-end próprio (ctypes cru) teria que
resolver essa negociação de GL do zero.

**Achado que muda o plano (evidência real, não suposição):** em vez de escrever esse front-end do
zero, a máquina de desenvolvimento já tem `retroarch` E `Xvfb` instalados, e o binário `retroarch`
instalado contém de verdade (via `strings`, não documentação) toda a interface de **comando de
rede** que precisaríamos construir: `network_cmd_enable`, `network_cmd_send()`, `FRAMEADVANCE`,
`SAVE_STATE`/`LOAD_STATE`(`_SLOT`), `PAUSE_TOGGLE`, `GET_STATUS`, e - o mais importante -
`READ_CORE_MEMORY %x`/`WRITE_CORE_MEMORY %x`. Isso é software maduro, já testado, que já sabe
negociar o contexto GL corretamente (é literalmente o que já roda a captura de gameplay normal
hoje) - **elimina o maior risco do plano** em vez de exigir resolvê-lo. `Xvfb` dá um display X real
(embora invisível) pro RetroArch rodar seu pipeline de GL normal sem monitor. Isso é o caminho
primário do M1; um front-end ctypes cru é o Plano B explícito, só se o comando de rede do RetroArch
provar insuficiente pra algo específico.

## Nota sobre TAS/yabause-rr (investigado e descartado como caminho, com evidência)

Existe um fork separado, `yabause-rr` (TASVideos), com formato de movie próprio (**YMV** -
mnemônicos de input por frame + metadados de config determinística: BIOS, região, versão de core;
`rerecordCount`). Um arquivo TAS é pequeno porque **não grava estado nenhum** - só a config inicial
+ inputs; a reprodução bit-exata (incluindo "aleatoriedade" de batalhas/drops, que em hardware
clássico é pseudo-aleatoriedade determinística, não entropia real) depende de sempre reproduzir
desde o mesmo boot a frio.

Isso **não se aplica a este caso de uso**: o worker não precisa reproduzir desde o boot, só
continuar uma sessão já em andamento a partir de um savestate - então a técnica YMV/TAS não traz
benefício aqui, e portar formato de um fork separado, mal documentado (a própria wiki do TASVideos
marca a página como incompleta) e com sinais de abandono não se justifica por ora.

Uma suposição inicial nessa investigação estava errada e foi corrigida com evidência: o `rand()`
sem seed usado no ruído do LFO do SCSP (`vendor/yabause/src/scsp.c:451,486,4752` - bug do Yabause
**upstream**, idêntico e não introduzido pelo fork yabassanshiro - confirmado por diff linha a
linha) não é não-determinístico entre execuções, como se pensou a princípio - não há nenhum
`srand()` alcançável no build libretro (o único `srand(time(NULL))` do código-fonte vive em
`dreamcast/yui.c`, um front-end de outra plataforma, não compilado pelo `Makefile` do libretro). Sem
seed, `rand()` se comporta como `srand(1)` por definição da norma C - sequência fixa e reproduzível
na mesma build/libc. O problema real que sobra é só de fidelidade (o ruído gerado não é o algoritmo
real do hardware Saturn), não de determinismo entre execuções na mesma máquina - e mesmo esse
resíduo não afeta o worker deste plano, que nunca depende de reproduzir desde o boot.

## Decisão de arquitetura

- **Ingestão**: continua sendo a fila de eventos do `portal_trace.jsonl` gerado pela gameplay real
  ao vivo - nada muda aqui, é consumo incremental (tail) do que já existe.
- **Avanço sob demanda**: uma *tool* que o agente/LLM chama explicitamente quando decide que precisa
  de mais frames pra desambiguar algo - decorre no tempo do agente, não no tempo real do jogo.
- A mesma "API" deve poder devolver outros tipos de dado sob demanda (não só PC/trace - também
  leitura de memória, e no futuro o que mais for necessário), então o desenho é um pequeno serviço
  com múltiplos endpoints, não uma função única de propósito estreito.

## Marcos (execução ainda não iniciada)

### M1 - Spike primário: RetroArch real headless (Xvfb) + comando de rede (ir/não-ir)
Sem escrever nenhum front-end libretro novo. Passos, todos com software já instalado:
- Sobe `Xvfb :99 -screen 0 1280x720x24 &` (display X invisível, real o suficiente pro RetroArch
  negociar GL normalmente).
- Roda `DISPLAY=:99 retroarch -L <yabause_libretro.so> <conteúdo> --appendconfig <cfg com
  network_cmd_enable=true, network_cmd_port=55355, video driver padrão>`, compilado com
  `PORTAL_TRACE=1` como já é feito hoje.
- Do lado do worker: um cliente UDP simples (`socket` puro, sem lib nova) mandando os comandos já
  confirmados no binário - `FRAMEADVANCE` (N vezes = N frames), `SAVE_STATE`/`LOAD_STATE_SLOT`,
  `READ_CORE_MEMORY <addr_hex> <len>`, `GET_STATUS`.
- **Critério de sucesso explícito:** consegue mandar `FRAMEADVANCE` várias vezes e confirmar (via
  `GET_STATUS` ou `READ_CORE_MEMORY` num endereço conhecido) que o estado emulado realmente avançou;
  `SAVE_STATE`+`LOAD_STATE_SLOT` faz um round-trip sem crash; o processo roda inteiramente sob
  `Xvfb`, sem precisar de display físico.
- **Plano B, só se M1 falhar de verdade** (ex.: `READ_CORE_MEMORY` não cobre o espaço de endereço
  que precisamos, ou o protocolo UDP se mostra muito limitado pro volume de dados do trace): cai
  pro front-end ctypes cru direto contra o `.so` (`emulator_worker/spike_headless.py`), usando
  OSMesa/EGL surfaceless pra responder `RETRO_ENVIRONMENT_SET_HW_RENDER` manualmente - mais
  trabalho, mas documentado aqui como caminho conhecido, não inventado na hora se for preciso.

### M2 - Serviço wrapper (só depois de M1 confirmado)
Um pequeno serviço FastAPI (`emulator_worker/server.py`, mesmo padrão que `july-engine` já usa) que
só traduz HTTP → os comandos UDP do RetroArch validados no M1 (não precisa reimplementar nada do
core em Python/C, só orquestrar):
- `POST /savestate/load {slot}` → `LOAD_STATE_SLOT`.
- `POST /advance {n_frames}` → N × `FRAMEADVANCE`, lendo o `PORTAL_TRACE_PATH` dessa sessão
  antes/depois pra devolver só as linhas novas do JSONL.
- `GET /memory?addr=&len=` → `READ_CORE_MEMORY`.
- `GET /status` → `GET_STATUS` (+ leitura do PC via memória, se `GET_STATUS` não expuser isso
  diretamente - confirmar no M1).

### M3 - Tools do agente (hle-studio)
Em `hle_studio/agent_tools.py` (ou um novo módulo `emulator_tools.py` importado por ele), 3 tools
pydantic-ai novas, chamando o serviço do M2 via `httpx`:
- `advance_emulator_frames(n: int) -> str` (trace novo).
- `read_emulator_memory(addr_hex: str, length: int) -> str`.
- `get_emulator_status() -> str`.
Documentar no system prompt do agente de porte QUANDO usar isso: especificamente quando
`MechanicalTranslator`/`disassemble_function` reporta um `JMP @Rn`/`BRAF Rn` sem alvo estático
resolvido (`note` não-vazio) - não como primeira opção, só quando a informação estática realmente
não é suficiente, mesmo espírito do "esgotar alternativas" já documentado em
`docs/native_port_playbook.md` do projeto Saturn.

### M4 - Consumo incremental do trace ao vivo
Adapta `Sh2RustTraceRanker`/`SelectTargets` (hoje um `grep` de arquivo fechado, one-shot) para um
modo `--follow`, que faz tail incremental do `portal_trace_play.jsonl` crescendo em tempo real
durante uma sessão de jogo ativa, disparando `draft` nos endereços novos conforme aparecem - sem
polling em loop apertado (usar `inotify`/watch de tamanho de arquivo com wakeup real, não
`sleep`+check).

## Riscos explícitos
- M1 (caminho primário) depende de `network_cmd_enable`/porta UDP estarem realmente acessíveis e do
  protocolo de comando aceitar os parâmetros que precisamos (`READ_CORE_MEMORY` pode ter um limite
  de tamanho por chamada, por exemplo) - confirmado que os comandos EXISTEM no binário via
  `strings`, mas não que o protocolo completo atende o volume de uso necessário; validar isso é o
  próprio objetivo do M1, não algo a assumir resolvido só porque a string existe.
- Se M1 primário falhar de verdade, cai pro Plano B (ctypes+OSMesa/EGL, mais trabalho) - não
  prometer M2-M4 até M1 confirmar viabilidade, seja pelo caminho primário ou pelo B.
- Sem um mecanismo de replay de input determinístico desde o boot, "avançar frames" sob demanda só
  ajuda pra código que não depende de NOVO input do jogador a partir daquele ponto (ex.: resolver
  um salto indireto cujo alvo já está decidido por estado interno) - para trechos que dependem de
  input humano contínuo, o avanço sob demanda não substitui a gravação real. Deixar isso explícito
  pro agente no prompt, não fingir que resolve todos os casos de ambiguidade.
- `retro_run()` não pausa/retoma o `PortalTraceToggleCapture()` automaticamente - o serviço do M2
  precisa garantir que a captura esteja ativa antes de cada `/advance`, já que hoje isso só liga por
  hotkey de teclado (`libretro.c:1019-1028`).

## Verificação
- M1: RetroArch sobe sob `Xvfb` sem erro, comandos UDP (`FRAMEADVANCE`, `SAVE_STATE`/
  `LOAD_STATE_SLOT`, `READ_CORE_MEMORY`, `GET_STATUS`) respondem e o estado emulado muda de verdade
  entre chamadas (confirmado por leitura de memória antes/depois, não só "não deu erro").
- M2: teste real via `curl` contra os 4 endpoints, comparando o trace devolvido por `/advance` com
  o formato já consumido por `Sh2RustTraceRanker` (mesmas chaves JSON).
- M3: uma chamada real do agente de porte usando `advance_emulator_frames` num caso conhecido de
  `JMP @Rn` não resolvido, confirmando que o alvo aparece no trace novo devolvido.
- M4: rodar `hle-studio draft --follow` em paralelo a uma sessão de jogo simulada (replay de um
  jsonl existente linha a linha com delay), confirmando que novos alvos disparam o pipeline sem
  esperar o arquivo terminar.

## Arquivos críticos
- `retroarch` + `Xvfb` (binários já confirmados instalados na máquina de desenvolvimento via
  `which`/`strings`) - caminho primário do M1, nenhum código C/ctypes novo necessário se funcionar.
- `vendor/yabassanshiro/yabause/src/libretro/libretro.c` (API a ser chamada via ctypes/FFI, só se
  cair no Plano B do M1)
- `vendor/yabassanshiro/yabause/src/portal_trace.c/.h` (formato do trace, ponto de integração)
- (novo, no projeto Saturn) `emulator_worker/server.py` (wrapper HTTP→UDP do M2), e só se
  necessário `emulator_worker/spike_headless.py` (Plano B)
- (aqui no hle-studio) `hle_studio/agent_tools.py`, `hle_studio/plugins/sh2_rust.py`
  (`Sh2RustTraceRanker` ganha modo `--follow`)
