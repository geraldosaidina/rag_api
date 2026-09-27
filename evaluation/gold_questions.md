# Conjunto de avaliação técnica — perguntas de referência

Versão 1.0. Corpus `pfc_corpus`. Língua das perguntas: português.

Este conjunto serve para uma avaliação posterior. Não demonstra, por si, nenhuma hipótese de investigação. As respostas esperadas são notas factuais, não respostas redigidas pelo assistente.

A evidência foi obtida por extracção de texto dos PDF e pelo catálogo SQLite. Não foi usada a recuperação do sistema, o reranking nem respostas geradas.

O PDF sem texto (`249225c1-cbfa-4fad-b073-a54a465186ba`) não entra como evidência.

Categorias: `direct_fact` (A), `semantic_paraphrase` (B), `topic_discovery` (C), `technology_methodology` (D), `metadata_constrained` (E), `unsupported` (F), `cross_document` (G).

## Q001 — direct_fact

**Pergunta.** Quem elaborou o projecto de videovigilância IP para a Avenida 25 de Setembro e em que ano foi apresentado?

**PFC esperado.** Belso Bento Langa, 2016 (`a160046c-7797-4752-84f0-0acbefc6b3ec`).

**Pontos factuais.** Autor Belso Bento Langa; apresentação em Junho de 2016 no ISUTC; objectivo de dimensionar videovigilância IP com fibra óptica nessa avenida.

**Páginas.** 1 (capa) e 18 (objectivo geral).

**Razão.** Autoria, ano e objectivo estão no mesmo relatório.

## Q002 — direct_fact

**Pergunta.** Qual é o objectivo geral do projecto de Edson Samuel Langa?

**PFC esperado.** Edson Samuel Langa, 2018 (`da25666f-1540-47a9-b336-8223490c8948`).

**Pontos factuais.** Dimensionar um sistema de identificação por radiofrequência como ferramenta de ensino e aprendizagem no ISUTC, para práticas laboratoriais.

**Páginas.** 15 (resumo) e 17 (objectivo geral).

**Razão.** O objectivo está formulado no desenho teórico e resumido no início do relatório.

## Q003 — direct_fact

**Pergunta.** Qual é o objectivo do projecto de Lectícia Maconne Aurélio Cuna sobre o Vale do Infulene?

**PFC esperado.** Lectícia Maconne Aurélio Cuna, 2024 (`1f40a684-4498-4a32-92f6-c548a2e43e3c`).

**Pontos factuais.** Propor um sistema integrado de gestão de culturas agrícolas, com enfoque no controlo de pragas no Vale do Infulene.

**Páginas.** 17 (resumo) e 22 (objectivo geral).

**Razão.** O caso de estudo e o objectivo estão explícitos no relatório.

## Q004 — direct_fact

**Pergunta.** O que pretende o projecto de Nereid da Josefa Maurício King?

**PFC esperado.** Nereid da Josefa Maurício King, 2019 (`788b6e42-6c73-4d67-aee8-b826bdea2498`).

**Pontos factuais.** Sistema automatizado para divulgar promoções de estabelecimentos comerciais na Cidade de Maputo; protótipo funcional de aplicação móvel.

**Páginas.** 15 (resumo) e 17 (objectivo geral).

**Razão.** Finalidade de um PFC identificado pela autora.

## Q005 — direct_fact

**Pergunta.** Qual é a finalidade do projecto de Arnaldo Manuel Nhaguilunguane no Centro de Saúde 1º de Maio?

**PFC esperado.** Arnaldo Manuel Nhaguilunguane, 2021 (`6601c5a5-beee-43ec-b654-4c3cb275e990`).

**Pontos factuais.** Acompanhamento de pacientes em tratamento prolongado; aplicação móvel e portal web; a capa chama ao conjunto ecossistema SaaS; entrevistas à equipa do centro em Abril de 2021.

**Páginas.** 1, 14 e 16.

**Razão.** Finalidade e caso de estudo estão na capa, no resumo e no objectivo geral.

## Q006 — semantic_paraphrase

**Pergunta.** Que projecto propõe etiquetas que identificam sozinhas objectos por sinais de rádio, para servir de recurso de laboratório a estudantes de telecomunicações?

**PFC esperado.** Edson Samuel Langa, 2018 (`da25666f-1540-47a9-b336-8223490c8948`).

**Pontos factuais.** O relatório dimensiona identificação por radiofrequência para ensino e prevê experiência laboratorial.

**Páginas.** 15 e 17.

**Terminologia do documento.** «Sistema de identificação por radiofrequência»; ferramenta de ensino; experiência laboratorial.

**Paráfrase da pergunta.** Etiquetas que identificam objectos por sinais de rádio, como recurso de laboratório.

**Razão.** A pergunta não usa RFID, radiofrequência nem o nome do autor. A recuperação lexical pelo título fica enfraquecida.

## Q007 — semantic_paraphrase

**Pergunta.** Que trabalho dimensiona uma rede sem fios de banda larga, com qualidade de serviço, para um distrito em que os moradores já têm voz móvel mas não recebem internet da operadora fixa?

**PFC esperado.** Mayra Patrícia Duarte Sofiano, 2015 (`5a49b09b-8393-4d24-bf99-f0817067972b`).

**Pontos factuais.** Rede de acesso WiMax para banda larga da TDM em Marracuene. O distrito tem voz móvel (mcel, Vodacom, Movitel) e não tem banda larga fixa.

**Páginas.** 16, 18 e 19.

**Terminologia do documento.** WiMax; TDM; distrito de Marracuene; internet banda larga.

**Paráfrase da pergunta.** Rede sem fios de banda larga com qualidade de serviço, num distrito com voz móvel e sem internet da operadora fixa.

**Razão.** A pergunta evita WiMax, Marracuene e TDM. O problema da página 18 distingue este trabalho da rede híbrida de Matola-Rio.

## Q008 — semantic_paraphrase

**Pergunta.** Como reforçar o sinal de telemóvel de terceira geração dentro de um prédio de casas e escritórios, quando as paredes e a cave impedem que o sinal das torres exteriores chegue com qualidade?

**PFC esperado.** Mércia Cecília David Fortes, 2018 (`cd669613-e54e-4b55-b452-4a3dd127ae2c`).

**Pontos factuais.** Sistema de antenas distribuídas para a cobertura interior da rede 3G da Vodacom no Edifício Platinum. A atenuação é atribuída às paredes, à cave e às macrocélulas exteriores.

**Páginas.** 17 e 19.

**Terminologia do documento.** Antenas distribuídas; cobertura indoor; rede 3G; Edifício Platinum; Vodacom.

**Paráfrase da pergunta.** Sinal de telemóvel de terceira geração dentro de um prédio misto, bloqueado por paredes, cave e distância às torres.

**Razão.** A pergunta não usa a sigla DAS, «antenas distribuídas», «indoor», Vodacom nem Platinum.

## Q009 — semantic_paraphrase

**Pergunta.** Que proposta troca o cabo de cobre da última milha por fibra que chega ao edifício ou ao armário junto à rua, sem equipamentos activos no meio do percurso?

**PFC esperado.** Felipe Eduardo Teixeira Viveros, 2011 (`d260a009-5aff-4023-b262-be3055c3c4cf`).

**Pontos factuais.** Rede óptica passiva EPON, com FTTB e FTTC, para substituir o cobre na Polana Cimento A.

**Páginas.** 13, 14 e 15.

**Terminologia do documento.** Redes ópticas passivas; EPON; FTTB e FTTC; cobre na rede de acesso.

**Paráfrase da pergunta.** Fibra até ao edifício ou ao armário de rua, sem equipamentos activos no percurso, no lugar do cobre da última milha.

**Razão.** A pergunta explica as siglas em linguagem corrente e não nomeia o bairro.

## Q010 — semantic_paraphrase

**Pergunta.** Que proposta permite pedir à distância o documento que comprova se a pessoa tem antecedentes, para não voltar à repartição a levantá-lo, quando esse papel é exigido para emprego ou carta de condução?

**PFC esperado.** Eugénia Laurícia Alcino Nhacuonga, 2023 (`620236cd-826a-4825-9c04-182064491f52`).

**Pontos factuais.** Protótipo para solicitar o certificado de registo criminal em Maputo. O documento prova antecedentes criminais e é exigido, entre outros fins, para carta de condução e concursos. O processo actual exige mais do que uma ida à repartição.

**Páginas.** 15, 16 e 17.

**Terminologia do documento.** Certificado de registo criminal; antecedentes criminais; deslocações repetidas à repartição.

**Paráfrase da pergunta.** Pedido à distância do documento de antecedentes, sem segunda ida à repartição, para emprego ou carta de condução.

**Razão.** A designação oficial do documento não entra na pergunta. Permanecem palavras da situação real descrita no texto, como antecedentes e carta de condução.

## Q011 — semantic_paraphrase

**Pergunta.** Que estudo procura falhas que um atacante possa explorar nos sítios web de ministérios e de outras entidades do Estado moçambicano?

**PFC esperado.** Vagner Flávio Fafetine Nhachungue, 2024 (`92ae9c58-ad57-4ccf-b648-d7b95bb31a40`).

**Pontos factuais.** Avaliação de vulnerabilidades em portais do governo, com WPScan, ZAP e Nuclei, classificadas pela OWASP Top 10 de 2021.

**Páginas.** 17 e 20.

**Terminologia do documento.** Avaliação de vulnerabilidades em portais do governo de Moçambique.

**Paráfrase da pergunta.** Falhas exploráveis em sítios web de ministérios e de outras entidades do Estado.

**Razão.** A pergunta não repete «vulnerabilidades» nem «portais do governo».

## Q012 — topic_discovery

**Pergunta.** Que projectos tratam de acesso sem fios à internet ou de melhorar a cobertura da rede móvel?

**PFC esperados (primários).** Mayra Sofiano, 2015; Diana Matsinhe, 2015; Mércia Fortes, 2018.

**Pontos factuais.** WiMax para banda larga em Marracuene; híbrido GPON–WiMax 802.16d em Matola-Rio; antenas distribuídas para cobertura interior 3G.

**Páginas.** Mayra 16 e 19; Diana 12 e 15; Mércia 17 e 19.

**Razão.** São os três trabalhos cujo objecto é acesso sem fios ou cobertura móvel. A EPON da Polana é fibra. A identificação por rádio do ISUTC é ensino, não cobertura celular.

## Q013 — topic_discovery

**Pergunta.** Que projectos propõem uma plataforma digital para pedir um documento oficial, recrutar jogadores ou financiar trabalhos académicos?

**PFC esperados (primários).** Eugénia Nhacuonga, 2023; Hassan Mutole, 2025; Vitilio Martins de Sousa Júnior, 2025.

**Pontos factuais.** Plataforma de certificado de registo criminal; plataforma de recrutamento da Liga Desportiva de Maputo; plataforma de financiamento de projectos académicos no ISUTC.

**Páginas.** Eugénia 15 e 17; Hassan 14 e 17; Vitilio 15 e 18.

**Razão.** A pergunta restringe os serviços. Outras aplicações do corpus não cobrem estes três casos.

## Q014 — topic_discovery

**Pergunta.** Que projectos dimensionam um sistema de câmaras para vigiar um espaço físico, seja uma avenida ou um condomínio?

**PFC esperados (primários).** Belso Langa, 2016; Óscar Guibunda, 2019.

**Pontos factuais.** Videovigilância IP com fibra na Avenida 25 de Setembro; vídeo-vigilância do condomínio Belo-Horizonte, simulada no IP Video System Design Tool.

**Páginas.** Belso 15 e 18; Óscar 14 e 16.

**Razão.** Só estes dois relatórios têm câmaras de um espaço físico como objecto do projecto.

## Q015 — topic_discovery

**Pergunta.** Que projectos adoptam a fibra óptica como meio de transmissão da solução que propõem?

**PFC esperados (primários).** Felipe Viveros, 2011; Diana Matsinhe, 2015; Belso Langa, 2016.

**Pontos factuais.** EPON com FTTB/FTTC; GPON na parte óptica da rede híbrida; fibra como meio da videovigilância IP.

**Páginas.** Felipe 13 e 14; Diana 12 e 15; Belso 15 e 18.

**Razão.** A fibra tem de ser o meio da solução proposta. Menções teóricas ou alternativas não foram marcadas como relevantes.

## Q016 — topic_discovery

**Pergunta.** Que projectos chegaram a construir um protótipo de software, para além de estudar o problema?

**PFC esperados (primários).** Nereid King, 2019; Eugénia Nhacuonga, 2023; Arnaldo Nhaguilunguane, 2021; Lectícia Cuna, 2024; Hassan Mutole, 2025; Vitilio Martins, 2025.

**Pontos factuais.** Cada resumo declara um protótipo: aplicação móvel de promoções; pedido de certificado; aplicação web e móvel de saúde; gestão agrícola; interfaces de recrutamento; plataforma de financiamento.

**Páginas.** Nereid 15; Eugénia 15 e 17; Arnaldo 14; Lectícia 17; Hassan 14; Vitilio 15.

**Razão.** Dimensionamento de rede, videovigilância, experiência de identificação por rádio e inspecção de sítios já existentes não apresentam este resultado.

## Q017 — technology_methodology

**Pergunta.** Que padrão de rede óptica e que formas de chegada da fibra são propostos para a Polana Cimento A?

**PFC esperado.** Felipe Viveros, 2011 (`d260a009-5aff-4023-b262-be3055c3c4cf`).

**Pontos factuais.** EPON; FTTB e FTTC; substituição do cobre na última milha.

**Páginas.** 13 e 14.

**Razão.** A arquitectura está nomeada no resumo e justificada na introdução.

## Q018 — technology_methodology

**Pergunta.** Que ferramentas de inspecção e que classificação de risco foram usadas no estudo das falhas de segurança dos sítios do governo?

**PFC esperado.** Vagner Nhachungue, 2024 (`92ae9c58-ad57-4ccf-b648-d7b95bb31a40`).

**Pontos factuais.** WPScan, ZAP e Nuclei; OWASP Top 10 de 2021; pesquisa exploratória e descritiva; riscos crítico, alto, médio e baixo.

**Páginas.** 17 e 20.

**Razão.** O resumo declara as ferramentas e a classificação usadas.

## Q019 — technology_methodology

**Pergunta.** Que técnica é proposta para identificar pragas e que abordagem de investigação foi usada no sistema agrícola do Vale do Infulene?

**PFC esperado.** Lectícia Cuna, 2024 (`1f40a684-4498-4a32-92f6-c548a2e43e3c`).

**Pontos factuais.** Identificação automatizada de pragas com inteligência artificial; investigação mista com inquéritos, entrevistas semiestruturadas e pesquisa documental; prototipagem e UML.

**Páginas.** 17 e 22.

**Razão.** Tecnologia e método estão no resumo do mesmo projecto cujo objectivo é o controlo de pragas.

## Q020 — technology_methodology

**Pergunta.** Que abordagem de investigação e que técnicas de modelação foram usadas no projecto de recrutamento de jogadores da Liga Desportiva de Maputo?

**PFC esperado.** Hassan Mutole, 2025 (`bcd14962-06cf-4ac4-bfd1-dd45f2332da1`).

**Pontos factuais.** Abordagem mista; entrevistas semiestruturadas a técnicos e dirigentes; UML e prototipagem de interfaces.

**Páginas.** 14 e 17.

**Razão.** O resumo descreve o procedimento, não apenas o tema.

## Q021 — metadata_constrained

**Pergunta.** Que projectos finais de curso deste arquivo são de 2015?

**PFC esperados (primários).** Mayra Sofiano e Diana Matsinhe, ambos de Maio de 2015.

**Pontos factuais.** Dois relatórios do ISUTC, licenciatura em Engenharia Informática e de Telecomunicações: WiMax em Marracuene e rede híbrida em Matola-Rio.

**Páginas.** Capa (página 1) de cada um.

**Razão.** O ano pedido coincide com o ano de capa e com o ano do catálogo. Não há terceiro PFC de 2015.

## Q022 — metadata_constrained

**Pergunta.** Qual é o projecto de Mércia Cecília David Fortes e em que ano foi apresentado?

**PFC esperado.** Mércia Cecília David Fortes, 2018 (`cd669613-e54e-4b55-b452-4a3dd127ae2c`).

**Pontos factuais.** Antenas distribuídas para a cobertura interior 3G da Vodacom no Edifício Platinum; Maio de 2018; supervisão de Eng. Nelson Mandava; ISUTC.

**Páginas.** 1 e 19.

**Razão.** A pergunta parte do nome da autora e pede o tema e o ano que a capa associa a esse nome.

## Q023 — metadata_constrained — sem suporte

**Pergunta.** Que projectos finais de curso deste arquivo foram realizados na Universidade Zambeze?

**PFC esperados.** Nenhum.

**Pontos factuais.** Não há evidência de PFC da Universidade Zambeze. Os projectos indexados estão atribuídos ao ISUTC.

**Páginas.** Nenhuma.

**Porque é sem suporte.** A cadeia «zambeze» não ocorre nos 14 PDF legíveis. O catálogo confirma a instituição das capas. O PDF ilegível não foi usado.

## Q024 — unsupported

**Pergunta.** Qual é o valor das propinas da licenciatura no ISUTC?

**PFC esperados.** Nenhum.

**Pontos factuais.** O corpus não contém o valor das propinas.

**Páginas.** Nenhuma.

**Porque é sem suporte.** Pesquisa integral de «propina» nos 14 PDF, sem ocorrências. Ser do ISUTC não implica tabelas de propinas.

## Q025 — unsupported

**Pergunta.** Que projectos finais de curso estudam a implantação de redes móveis de quinta geração em Moçambique?

**PFC esperados.** Nenhum.

**Pontos factuais.** Nenhum relatório estuda 5G. Existem trabalhos de 3G e de WiMax, que são outra geração e outra tecnologia.

**Páginas.** Nenhuma.

**Porque é sem suporte.** Pesquisa de «quinta geração» e da sigla 5G como palavra, sem ocorrências. Sequências «5g» dentro de GHz ou Gbps não foram tratadas como estudo de quinta geração.

## Q026 — unsupported

**Pergunta.** Que projectos finais de curso de 2020 estão neste arquivo?

**PFC esperados.** Nenhum.

**Pontos factuais.** Nenhum PFC indexado é de 2020. Os anos existentes são 2011, 2015, 2016, 2018, 2019, 2021, 2023, 2024 e 2025.

**Páginas.** Nenhuma.

**Porque é sem suporte.** O catálogo não tem ano 2020. As ocorrências do numeral 2020 no texto são citações, estatísticas ou códigos de equipamento, não o ano de um projecto deste arquivo.

## Q027 — cross_document

**Pergunta.** Compare a videovigilância proposta para a Avenida 25 de Setembro com a proposta para o condomínio Belo-Horizonte: que espaço cada uma vigia e que meio ou ferramenta de projecto utiliza?

**PFC esperados (primários).** Belso Langa, 2016, e Óscar Guibunda, 2019. Os dois são necessários.

**Pontos factuais.** Avenida pública com videovigilância IP sobre fibra, face a condomínio residencial diagnosticado por entrevistas e simulado no IP Video System Design Tool 9.1.

**Páginas.** Belso 15 e 18; Óscar 14 e 16.

**Razão.** Os dois dimensionam câmaras e diferem no espaço e na forma de sustentar o projecto.

## Q028 — cross_document

**Pergunta.** Compare as redes de acesso que Diana João Matsinhe e Mayra Patrícia Duarte Sofiano dimensionam para a TDM: que tecnologias usa cada uma e para que zona?

**PFC esperados (primários).** Diana Matsinhe, 2015, e Mayra Sofiano, 2015. Os dois são necessários.

**Pontos factuais.** Matola-Rio com GPON e WiMax 802.16d, face a Marracuene com rede de acesso WiMax simulada no Radio Mobile.

**Páginas.** Diana 12 e 15; Mayra 16 e 19.

**Razão.** São dois dimensionamentos da mesma operadora, com arquitectura e geografia distintas.
