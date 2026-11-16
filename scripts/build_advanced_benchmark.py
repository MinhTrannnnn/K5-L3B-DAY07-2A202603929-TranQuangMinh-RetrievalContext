"""Author and freeze a disjoint development/test benchmark from original WDR pages."""
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf
from src.common import ROOT, digest, write_json


def main():
    target = ROOT / 'benchmark/advanced_cases.json'
    if target.exists():
        raise FileExistsError('Benchmark already frozen. Use a new version for a new experiment.')
    cases = []

    def add(split, group, question, answer, evidence):
        identity = ('D' if split == 'dev' else 'T') + f'{sum(c["split"] == split for c in cases) + 1:02}'
        groups = []
        for page, label, description in evidence:
            groups.append([{'source_id': 'wdr', 'page': str(page), 'printed_page': str(page - 32),
                            'region_label': label, 'evidence_description': description,
                            'bbox': list(document[page - 1].rect)}])
        cases.append({'id': identity, 'split': split, 'group': group, 'question': question,
                      'answerable': True, 'expected_answer': answer, 'evidence_groups': groups,
                      'author': 'Codex assistant, original PDF inspected',
                      'human_confirmation': 'pending'})

    def absent(split, question, reason, terms):
        identity = ('D' if split == 'dev' else 'T') + f'{sum(c["split"] == split for c in cases) + 1:02}'
        cases.append({'id': identity, 'split': split, 'group': 'missing', 'question': question,
                      'answerable': False, 'expected_behavior': 'search_and_abstain',
                      'expected_answer': 'The report does not provide the requested information.',
                      'absence_reason': reason, 'absence_search_terms': terms,
                      'absence_check': 'Full-document text search plus relevant section inspection; confirm against original PDF.',
                      'human_confirmation': 'pending'})

    with pymupdf.open(ROOT / 'data/documents/WDR2024.pdf') as document:
        add('dev','table','Inspect Table O.2. Is investment a higher or lower priority for each of the three income groups?',
            'Investment is a higher priority for low-, lower-middle- and upper-middle-income countries.',
            [(39,'Table O.2','Investment column has Higher priority in all three rows.')])
        add('dev','table','In Table O.3, what is the enterprise policy for low-income countries?',
            'Improve the investment climate to increase domestic and foreign investment.',
            [(59,'Table O.3','Enterprise row, low-income column.')])
        add('dev','table','Inspect the original rotated Table S.1. What GNI per capita in current US dollars (2023) is listed for India?',
            'US$2,540, Atlas method, current US dollars in 2023.',
            [(81,'Table S.1','India row and GNI per capita column, value 2,540.')])
        add('dev','image','Read the legend of Map 6.1. Which income group is colored orange?',
            'Upper-middle-income countries.',[(173,'Map 6.1','Income group legend, orange swatch.')])
        add('dev','image','Inspect Map 6.3. Where in China are low-carbon technology jobs concentrated compared with fossil-fuel jobs?',
            'Low-carbon jobs cluster in southeast-coast manufacturing hubs, while fossil-fuel jobs are near inland coal mines.',
            [(186,'Map 6.3','Two maps comparing the spatial distribution of low-carbon and fossil-fuel jobs.')])
        add('dev','chart','Inspect Figure O.1. Around 2022, does Korea or the median middle-income country have the higher GNI per capita relative to the United States?',
            'Korea has the higher relative GNI per capita.',[(35,'Figure O.1','Korea line is much higher than the median middle-income line at the right endpoint.')])
        add('dev','chart','Read Figure O.2. What values are shown on the two bars for GDP per worker and physical/human capital relative to the United States?',
            'GDP per worker is 21 percent; physical and human capital is 71 percent.',
            [(37,'Figure O.2','Bars labeled 21 and 71, percent.')])
        add('dev','chart','In Figure O.5, which of Korea, Poland and Chile has the highest GNI per capita at the right endpoint?',
            'The Republic of Korea.',[(40,'Figure O.5','Korea is the top line at the right endpoint, above Poland and Chile.')])
        add('dev','chart','Read Figure 5.8. Which two regions are highlighted as having low female labor force participation?',
            'Middle East and North Africa, and South Asia.',[(156,'Figure 5.8','Regional line plot and legend for female labor force participation, ages 15–64.')])
        add('dev','text','According to the overview, how many middle-income countries are discussed, and what per-capita income range defines them?',
            '108 countries, with per-capita income between US$1,136 and US$13,845.',[(33,'In brief','First paragraph gives 108 and the income range.')])
        add('dev','text','What year does the report give for India’s ambition to become a developed economy?',
            '2047, the centennial of independence.',[(34,'To get rich is glorious','Paragraph on national ambitions states India 2047.')])
        add('dev','text','What does China’s 14th Five-Year Plan envision for 2035, as described in the overview?',
            'Reaching the median GDP per capita of developed nations.',[(34,'To get rich is glorious','Paragraph on China’s 2035 ambition.')])
        add('dev','text','Approximately how many people live in the middle-income countries discussed by the report?',
            '6 billion people.',[(33,'In brief','The text refers to 6 billion people in today’s middle-income countries.')])
        add('dev','text','Name the domestic demographic and debt pressures narrowing the prospects of middle-income economies in the overview.',
            'Rising debt and aging populations.',[(33,'In brief','During the last decade paragraph lists rising debt and aging populations at home.')])
        add('dev','text','Why can crises make it easier to dismantle outdated arrangements, according to the overview?',
            'In democracies they can help forge the consensus needed for tough policy reforms.',
            [(34,'Overview continuation','First paragraph explains crises and consensus for reforms.')])
        absent('dev','What exact GDP per capita does the report predict for India on 1 January 2099?',
               'No dated point forecast for India on that exact day.', ['2099','1 January 2099'])
        absent('dev','What is the report’s prescribed universal interest rate that guarantees innovation in every country?',
               'No universal guaranteed interest rate is prescribed.', ['universal interest rate','guarantees innovation'])
        absent('dev','What password does the World Bank require to access the confidential underlying firm-level data?',
               'The public report contains no data-access password.', ['password','confidential'])

        add('test','table','Read Table O.1. What percentage of the global population is in lower-middle-income countries in 2022?',
            '40.3 percent.',[(36,'Table O.1','Lower-middle-income row, share of global population column: 40.3.')])
        add('test','table','In Table O.1, what share of global carbon dioxide emissions is attributed to upper-middle-income countries in 2022?',
            '48.6 percent.',[(36,'Table O.1','Upper-middle-income row, CO2 emissions column: 48.6.')])
        add('test','table','According to Table O.1, what percentage of global GDP is produced by high-income countries in 2022?',
            '60.8 percent.',[(36,'Table O.1','High-income row, share of global GDP: 60.8.')])
        add('test','table','In Table O.1, what share of people in extreme poverty globally is in low-income countries?',
            '36.5 percent.',[(36,'Table O.1','Low-income row, extreme poverty share: 36.5.')])
        add('test','table','Inspect Table 4.1. What productivity benefit and drawback can market power have?',
            'Mergers and acquisitions can improve efficiency; conglomerates can cross-subsidize unproductive firms and crowd out more productive competitors.',
            [(135,'Table 4.1','Productivity row, pros and cons columns.')])
        add('test','table','Read the consumer welfare row of Table 4.1. What are the stated pro and con?',
            'More efficient structures can lower costs and prices; fewer market players make cartels easier to form.',
            [(135,'Table 4.1','Consumer welfare row, pros and cons columns.')])
        add('test','image','Inspect Figure O.8. Match the Energy, Talent and Enterprise panels to the economists named above them.',
            'Energy: Schumpeter; Talent: Aghion and Howitt; Enterprise: Akcigit and Kerr.',
            [(44,'Figure O.8','Three illustrated panels with the field and economist names.')])
        add('test','image','Inspect Figure O.10. Which force is oversized in the left-hand configuration, and how are the three forces rebalanced on the right?',
            'Preservation is oversized on the left. On the right preservation shrinks while creation and destruction become stronger/more balanced.',
            [(49,'Figure O.10','Before/after circle diagram shows a large orange Preservation circle becoming smaller and the other two becoming larger.')])
        add('test','image','In the quality-ladder schematic Figure 4.3, which firm is marked IN and which is marked OUT?',
            'Firm 5 is IN and Firm 4 is OUT.',[(120,'Figure 4.3','Top ladder rung has incoming Firm 5; next rung has outgoing Firm 4.')])
        add('test','image','Inspect Figure 7.2. What box is directly above Contestability, and which three policy boxes feed into Contestability?',
            'Disciplining incumbency is above Contestability; Institutions, Incentives and Interventions feed into it.',
            [(195,'Figure 7.2','Arrow diagram with three base boxes, central Contestability and top Disciplining incumbency.')])
        add('test','image','Read Map 6.2. What do the orange lines represent, and what year do the mapped networks describe?',
            'Power transmission lines; electricity transmission networks in 2023.',
            [(182,'Map 6.2','Map legend labels orange lines as power transmission lines; note specifies networks in 2023.')])
        add('test','chart','Inspect Figure 3.1. In 2020, is the harmful-trade-policy line above or below the helpful-policy line, and what does the vertical axis count?',
            'The harmful line is above the helpful line. The y-axis counts new global trade policies.',
            [(104,'Figure 3.1','Orange harmful line above dark helpful line at 2020; y-axis is Number of new global trade policies.')])
        add('test','chart','Read the legend and vertical axes of Figure 3.2. Which two semiconductor HS codes are plotted, and what is counted?',
            '8541 and 8542 semiconductors; the number of new trade interventions per year.',
            [(105,'Figure 3.2','Legend lists 8541/8542; panels count new helpful/harmful interventions per year.')])
        add('test','chart','Inspect Figure 6.1. Is the vertical price axis logarithmic or linear, and by about 2020 are solar photovoltaics lower or higher than software on that index?',
            'Logarithmic; solar photovoltaics are lower than software.',
            [(171,'Figure 6.1','Axis explicitly says log scale; PV curve ends far below the software curve.')])
        add('test','chart','Read Figure 7.8. What shares are shown for SOEs in coal capacity and the private sector in renewable capacity, and what is their percentage-point difference?',
            '84 percent for SOEs in coal and 80 percent for private renewable capacity; difference 4 percentage points. China is shown separately.',
            [(201,'Figure 7.8','Two pie charts show SOEs 84% coal and private sector 80% renewable; note separates China ownership.')])
        add('test','chart','Inspect Figure 9.12. What unit is used for global primary energy consumption, and what is the final year on the horizontal axis?',
            'Terawatt-hours (TWh); 2022.',[(267,'Figure 9.12','Stacked area chart y-axis includes TWh, x-axis ends in 2022.')])
        add('test','multi_hop','Using Table O.1 and Figure O.4, calculate the combined global population share of lower- and upper-middle-income countries and identify the 2i and 3i strategy components.',
            '40.3 + 35.1 = 75.4 percent. 2i is investment plus infusion; 3i adds innovation.',
            [(36,'Table O.1','Population shares 40.3 and 35.1.'),(39,'Figure O.4','2i Investment + Infusion; 3i Investment + Infusion + Innovation.')])
        add('test','multi_hop','Combine Table O.2 with Table O.3: is infusion a high priority for lower-middle-income countries, and how should their enterprise policy discipline market leaders?',
            'Yes, infusion is a higher priority; discipline market leaders through integration into globally contestable markets.',
            [(39,'Table O.2','Lower-middle-income row, infusion higher priority.'),(59,'Table O.3','Lower-middle-income enterprise column recommends globally contestable markets.')])
        add('test','multi_hop','Using Figure 7.8 and the Energy row of Table O.3, state the SOE share in coal capacity and the recommended budget-constraint policy for SOEs in lower-middle-income countries.',
            'SOEs have 84 percent of coal capacity in the depicted sample; harden SOE budget constraints.',
            [(201,'Figure 7.8','Coal pie chart SOEs 84%, China separate.'),(59,'Table O.3','Lower-middle-income Energy row: discipline SOEs by hardening budget constraints.')])
        add('test','multi_hop','Combine the Jobs row of Table 4.1 with the foundational-skills discussion on printed page 190. How can demand for skilled workers rise while educational preparation remains a constraint?',
            'Market power can raise demand for highly skilled workers while depressing low-skilled wages/capturing talent. The text says upper-secondary enrollment/completion lag and weak foundational skills weaken the talent pipeline.',
            [(135,'Table 4.1','Jobs row pros and cons.'),(222,'Foundational skills','Upper-secondary enrollment/completion lag; unless foundational skills improve the talent pipeline remains weak.')])
        add('test','text','How does the structural-change section define the middle-income trap?',
            'Risk of slowdown or stagnation when a country fails to adapt policies and institutions to changing economic and structural needs.',
            [(87,'Economic development = structural change','First paragraph defines middle-income trap.')])
        add('test','text','According to the overview, what high-income transition year and GDP-per-capita growth ambition does Viet Nam’s 2021–2030 strategy specify?',
            'Transition to high-income status by 2045, with sustained GDP per capita growth of 7 percent through the decade.',
            [(34,'To get rich is glorious','Viet Nam paragraph gives 7 percent and 2045.')])
        add('test','text','How many new low-carbon-technology job openings were identified in 2022, and approximately what share of online jobs did they represent?',
            '3.3 million; just under 2 percent of jobs posted online in 2022.',
            [(171,'Low-carbon jobs paragraph','Paragraph on one billion postings across 86 countries states 3.3 million and just under 2 percent.')])
        add('test','text','What is one terawatt-hour equivalent to in watt-hours, according to the note beneath Figure 9.12?',
            'One trillion watt-hours.',[(267,'Figure 9.12 note','Note defines one TWh as 1 trillion watt-hours.')])
        absent('test','What exact 2037 GNI per capita does Table O.1 give for Viet Nam?',
               'Table O.1 describes 2022 income groups and has no 2037 Viet Nam forecast.', ['2037','Table O.1'])
        absent('test','What exact percentage of global population will be upper-middle-income in 2099 according to this report?',
               'The requested 2099 point estimate is not supplied.', ['2099','upper-middle-income'])
        absent('test','What exact US-dollar tariff for one household’s electricity does Map 6.2 prescribe?',
               'The map depicts transmission networks and supplies no household tariff.', ['household tariff','Map 6.2'])
        absent('test','What is the guaranteed stock-market return from investing in every company mentioned in this report?',
               'No guaranteed stock return is provided for the companies mentioned.', ['guaranteed stock','stock-market return'])
        absent('test','What are the names and personal phone numbers of every job applicant represented in Map 6.1?',
               'The map is aggregate geographic job-posting data, not individual applicants or phone numbers.', ['phone numbers','applicant'])
        absent('test','What exact price and launch date does this report predict for a commercial fusion reactor in 2049?',
               'The report does not give this exact product price and launch-date prediction.', ['2049','fusion reactor'])

    write_json(target, cases)
    write_json(ROOT / 'benchmark/advanced_freeze.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'cases_sha256': digest(target.read_bytes()),
        'pdf_sha256': digest((ROOT/'data/documents/WDR2024.pdf').read_bytes()),
        'split_counts': dict(Counter(case['split'] for case in cases)),
        'test_groups': dict(Counter(case['group'] for case in cases if case['split']=='test')),
        'policy': 'Questions frozen before retrieval experiments. Do not tune on test. Gold only used in evaluation.',
        'human_confirmation': 'pending'})
    print(len(cases), 'cases frozen')


if __name__ == '__main__':
    main()
