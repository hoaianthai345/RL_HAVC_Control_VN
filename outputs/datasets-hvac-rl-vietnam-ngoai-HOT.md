# Research nhanh: dataset phù hợp để train HVAC RL ở Việt Nam (ngoài HOT)

## Kết luận ngắn

Tôi **không tìm thấy một open dataset công khai ở Việt Nam** có đủ bộ `state + action + reward + HVAC control commands` ở mức tốt như các benchmark RL quốc tế hoặc như dataset HOT mà bạn muốn loại ra.

Vì vậy, nếu mục tiêu là train RL cho HVAC ở Việt Nam, hướng thực tế nhất là dùng **hybrid stack**:

1. **Dữ liệu Việt Nam** để localize khí hậu, hành vi sử dụng, giá điện, occupancy.
2. **Dataset khu vực nóng ẩm / dataset RL-ready quốc tế** để pretrain hoặc học offline RL.
3. **Simulator** (BOPTEST / Sinergym / CityLearn) gắn **weather VN + tariff VN** để fine-tune và evaluate.

---

## Đánh giá theo mức độ phù hợp cho HVAC RL ở Việt Nam

### Tier A — Nên dùng ngay trong pipeline

#### 1) Hanoi 49 apartments dataset (2024) — **local Vietnam, rất đáng dùng**
- **Tên**: *15-minute measurement datasets from 49 residential apartments in hot and humid climates for hygrothermal building simulation and occupant behavior studies*
- **DOI**: 10.17632/s9wkdww94w.2
- **Loại**: dữ liệu thực đo ở Hà Nội
- **Có gì**:
  - indoor temperature, humidity, pressure, CO2
  - weather ngoài trời: temperature, humidity, solar radiation, wind, rainfall
  - occupancy và AC operation suy diễn từ dữ liệu indoor
  - 15 phút trong 1 năm, 49 căn hộ
- **Điểm mạnh cho RL**:
  - đúng bối cảnh **Việt Nam, khí hậu nóng ẩm**
  - tốt để học occupancy model, comfort proxy, AC usage policy prior, reward shaping
- **Điểm yếu**:
  - chủ yếu là **residential split AC**, không phải BMS/central HVAC
  - action/control command không đầy đủ như offline RL benchmark chuẩn
- **Use case phù hợp**:
  - imitation/pretraining cho behavior model
  - demand/comfort model
  - calibrate simulator cho nhà ở Việt Nam
- **Mức khuyến nghị**: **Rất cao** nếu bài toán của bạn là residential/light commercial ở VN.

#### 2) EnergyPlus/EPW weather files cho Việt Nam — **bắt buộc nếu train simulator-based RL**
- **Nguồn**:
  - EnergyPlus weather catalog
  - Climate.OneBuilding
- **Có gì**:
  - EPW/TMY cho nhiều location, có thể lấy Hà Nội, TP.HCM, Đà Nẵng... nếu có station tương ứng
- **Điểm mạnh**:
  - gần như bắt buộc khi localize BOPTEST / Sinergym / CityLearn sang Việt Nam
- **Điểm yếu**:
  - không phải HVAC operation dataset; chỉ là climate driver
- **Use case**:
  - thay weather file của benchmark bằng khí hậu Việt Nam
- **Mức khuyến nghị**: **Bắt buộc** trong mọi pipeline RL dùng simulation.

#### 3) Dữ liệu giá điện Việt Nam (2025) — **hữu ích cho reward/cost**
- **Tên**: *Electricity prices and household electricity consumption in Vietnam*
- **DOI**: 10.17632/vccdzs9k7f.1
- **Có gì**: giá điện và tiêu thụ điện hộ gia đình Việt Nam
- **Điểm mạnh**:
  - giúp xây reward theo cost thay vì chỉ energy
  - hữu ích cho demand response / cost-aware RL
- **Điểm yếu**:
  - metadata công khai khá ít; chưa xác minh sâu cấu trúc trường dữ liệu
  - household focus, không phải HVAC command-level
- **Mức khuyến nghị**: **Trung bình-cao** để bổ sung reward/economic layer.

---

### Tier B — Dùng làm proxy gần Việt Nam hoặc dữ liệu giàu tín hiệu HVAC

#### 4) CU-BEMS (Thailand, 2020) — **proxy khu vực tốt nhất tôi tìm thấy**
- **Tên**: *CU-BEMS, smart building electricity consumption and indoor environmental sensor datasets*
- **DOI**: 10.1038/s41597-020-00582-3
- **Loại**: office building ở Bangkok
- **Có gì**:
  - 55 individual AC units
  - lighting + plug loads ở 33 zones
  - indoor temperature, humidity, ambient light ở 24 vị trí
  - 1-minute resolution, 2018–2019
- **Điểm mạnh**:
  - khí hậu Bangkok khá gần phần lớn miền Nam/miền Trung Việt Nam hơn dữ liệu US/EU
  - có granularity cao, hữu ích cho control-oriented modeling
- **Điểm yếu**:
  - không phải dataset RL chuẩn có action/reward đóng gói sẵn
  - office building Thái Lan, không phải Việt Nam
- **Use case**:
  - pretrain dynamics model / occupancy-response model
  - học surrogate model trước khi transfer sang VN
- **Mức khuyến nghị**: **Rất cao** nếu thiếu dữ liệu commercial VN.

#### 5) BEAR-Data (UCSD, 2023) — **offline RL-friendly**
- **Tên**: *BEAR-Data: Analysis and Applications of an Open Multizone Building Dataset*
- **Dạng public page**: Hugging Face + project homepage
- **Có gì**:
  - 1 building, 80+ zones, 17+ action commands
  - zone temperature, occupied status
  - damper commands, cooling/heating setpoints, flow setpoints
  - building energy load / power
- **Điểm mạnh**:
  - gần với bài toán RL điều khiển HVAC thật hơn nhiều dataset energy thông thường
  - có action signals rõ
- **Điểm yếu**:
  - không ở Việt Nam
  - climate/system khác
- **Mức khuyến nghị**: **Cao** cho offline RL pretraining hoặc benchmark thuật toán.

#### 6) Google Smart Buildings / SBSim (2024) — **RL-ready mạnh nhưng nặng**
- **Tên repo**: `google/sbsim`
- **Dataset**: `smart_buildings` trên TensorFlow Datasets
- **Có gì**:
  - 6 năm dữ liệu telemetric từ 3 office buildings
  - action / observation / reward structure rõ cho RL
  - download size ~11 GiB, dataset size ~86.77 GiB cho config hiển thị
- **Điểm mạnh**:
  - rất hợp cho offline RL benchmark
  - có calibrated simulation suite đi kèm
- **Điểm yếu**:
  - không phải Việt Nam
  - khá nặng, schema phức tạp
- **Mức khuyến nghị**: **Cao** nếu bạn muốn benchmark thuật toán offline RL nghiêm túc.

#### 7) B2RL (2022) — **benchmark offline RL chuyên cho building**
- **Tên**: *B2RL: an open-source dataset for building batch reinforcement learning*
- **arXiv**: 2209.15626
- **Có gì**:
  - real building buffer + simulated building buffer
  - state/action/reward mô tả rõ cho batch RL
  - real buffer khoảng 170k–260k datapoints, 15 rooms, 3 floors, ~1 năm
- **Điểm mạnh**:
  - thiết kế riêng cho offline RL trong building control
  - dễ dùng để benchmark thuật toán
- **Điểm yếu**:
  - không phải Việt Nam
  - độ đại diện cho khí hậu VN thấp hơn CU-BEMS
- **Mức khuyến nghị**: **Cao** cho benchmark; **trung bình** cho direct transfer sang VN.

---

### Tier C — Dùng để xây environment/simulator local hóa cho Việt Nam

#### 8) CityLearn
- **Loại**: RL environment + dataset schema cho community/building energy control
- **Điểm quan trọng**:
  - schema hỗ trợ building data, weather, carbon intensity, pricing
  - có thể thay bằng **weather VN + giá điện VN + load nội suy/local data**
- **Mức khuyến nghị**: **Rất cao** nếu bạn muốn dựng benchmark nhiều tòa nhà ở Việt Nam.

#### 9) Sinergym
- **Loại**: Gymnasium interface cho EnergyPlus control
- **Điểm quan trọng**:
  - hỗ trợ benchmark HVAC RL với EnergyPlus
  - có thể custom EPW Việt Nam và building model
- **Mức khuyến nghị**: **Rất cao** để fine-tune policy trong simulation.

#### 10) BOPTEST / BOPTEST-Gym
- **Loại**: benchmark framework cho building control
- **Điểm quan trọng**:
  - API chuẩn để so sánh control strategy
  - phù hợp để evaluate policy trước deploy
- **Điểm yếu**:
  - localize sang đúng building Việt Nam cần effort modelica/model mapping
- **Mức khuyến nghị**: **Cao** cho evaluation chuẩn hóa.

---

### Tier D — Dữ liệu Việt Nam bổ trợ nhưng không đủ để train RL trực tiếp

#### 11) Survey of housing, energy consumption and refurbishment in Vietnam (2020)
- **DOI**: 10.4121/13109924.v1
- **Giá trị**:
  - hỗ trợ prior về housing stock, retrofit, energy-use context ở Việt Nam
- **Hạn chế**:
  - survey dataset, không phải time-series HVAC control
- **Mức khuyến nghị**: **Trung bình** cho domain understanding; thấp cho direct RL training.

#### 12) USAID Vietnam Building Energy Performance Survey
- **Nguồn web search** cho thấy có dataset/catalog về building stock và building energy performance ở 5 thành phố lớn
- **Giá trị tiềm năng**:
  - useful cho building stock characterization, sampling tòa nhà, baseline consumption
- **Trạng thái xác minh**:
  - **unverified access**: trang catalog xuất hiện trong search, nhưng truy cập/fetch trực tiếp từ môi trường hiện tại trả 404
- **Mức khuyến nghị**: **Có tiềm năng**, nhưng cần bạn tự tải/xác minh thủ công.

---

## Ranking thực dụng cho bài toán HVAC RL ở Việt Nam

### Nếu bạn muốn làm **residential / split AC RL ở Việt Nam**
1. Hanoi 49 apartments
2. EPW weather Việt Nam
3. Electricity prices & household electricity consumption in Vietnam
4. CU-BEMS (proxy)
5. Sinergym / CityLearn để build simulator

### Nếu bạn muốn làm **commercial building / office HVAC RL ở Việt Nam**
1. CU-BEMS
2. EPW weather Việt Nam
3. USAID Vietnam BEPS (nếu lấy được raw data)
4. Google Smart Buildings / SBSim
5. BEAR-Data
6. BOPTEST / Sinergym để evaluate

### Nếu bạn muốn làm **offline RL benchmark trước, rồi transfer sang Việt Nam sau**
1. Google Smart Buildings / SBSim
2. B2RL
3. BEAR-Data
4. CU-BEMS
5. local Vietnam weather + tariff + occupancy priors

---

## Pipeline tôi khuyến nghị

### Phương án tốt nhất hiện tại

#### Option A — Thực dụng nhất
- **Dynamics / occupancy / comfort prior**: Hanoi 49 apartments + CU-BEMS
- **Reward cost**: giá điện Việt Nam
- **Climate**: EPW Hà Nội / TP.HCM / Đà Nẵng
- **Simulator**: Sinergym hoặc CityLearn
- **Evaluation**: BOPTEST hoặc Sinergym custom case

#### Option B — Nếu bạn muốn offline RL “đúng bài” hơn
- **Pretrain offline RL**: B2RL hoặc SBSim
- **Domain adaptation**: retrain dynamics bằng Hanoi/CU-BEMS
- **Fine-tune in sim**: Sinergym với EPW Việt Nam
- **Policy validation**: BOPTEST / custom EnergyPlus case

---

## Khoảng trống dữ liệu tôi thấy rõ

### Observation
- Dữ liệu Việt Nam công khai hiện có thiên về:
  - indoor environment
  - occupancy behavior
  - survey/building stock
  - weather / tariff
- Nhưng còn thiếu mạnh ở phần:
  - HVAC control commands/setpoints
  - BAS/BMS actuator logs
  - zone-level energy + command + comfort đồng thời
  - reward-ready trajectories cho offline RL

### Inference
- Nếu bạn cần một dataset **thay thế HOT để train RL trực tiếp end-to-end**, hiện tại **chưa có lựa chọn open tại Việt Nam mạnh tương đương** mà tôi xác minh được.
- Hướng khả thi hơn là **ghép nhiều dataset** để tạo training stack.

---

## Đề xuất ngắn gọn cho bạn

Nếu bạn muốn bắt đầu ngay, tôi khuyên chốt bộ sau:

1. **Hanoi 49 apartments** — local Vietnam core dataset  
2. **CU-BEMS** — commercial proxy giàu tín hiệu HVAC  
3. **Climate.OneBuilding / EnergyPlus EPW cho Việt Nam** — local weather  
4. **Electricity prices and household electricity consumption in Vietnam** — cost/reward  
5. **Sinergym hoặc CityLearn** — môi trường train  
6. **B2RL hoặc SBSim** — pretrain/offline RL benchmark  

---

## Blocked / Unverified

- **USAID Vietnam Building Energy Performance Survey**: tìm thấy qua web search nhưng **không verify được raw page/data access** trong môi trường hiện tại do URL fetch trả 404.
- **Electricity prices and household electricity consumption in Vietnam**: xác minh được trang dataset, nhưng chưa inspect sâu file schema thực tế.

---

## Sources

### Vietnam / local or near-local
- Hanoi 49 apartments dataset: https://data.mendeley.com/datasets/s9wkdww94w/2
- Vietnam housing survey dataset: https://data.4tu.nl/datasets/99fcb332-5ff0-4ef4-b5e7-9799e1ee3756
- Vietnam electricity price/consumption dataset: https://data.mendeley.com/datasets/vccdzs9k7f
- USAID/Vietnam BEPS catalog: https://catalog.data.gov/dataset/building-energy-performance-survey-usaid-vietnam-clean-energy-program
- CU-BEMS paper: https://doi.org/10.1038/s41597-020-00582-3
- CU-BEMS article page: https://www.nature.com/articles/s41597-020-00582-3

### Weather / simulation localization
- EnergyPlus weather catalog: https://openei.org/wiki/EnergyPlus_Weather_Data
- Climate.OneBuilding EPW repository: https://climate.onebuilding.org/

### RL-ready datasets / environments
- BOPTEST: https://ibpsa.github.io/project1-boptest/
- BOPTEST-Gym: https://github.com/ibpsa/project1-boptest-gym
- CityLearn docs: https://www.citylearn.net/
- CityLearn dataset docs: https://www.citylearn.net/overview/dataset.html
- Sinergym docs: https://ugr-sail.github.io/sinergym/compilation/main/index.html
- B2RL GitHub: https://github.com/HYDesmondLiu/B2RL
- B2RL arXiv: https://arxiv.org/abs/2209.15626
- BEAR project: https://ucsdsmartbuilding.github.io/
- Bear-Data: https://huggingface.co/datasets/alwaysbyx/Bear-Data
- Google SBSim: https://github.com/google/sbsim
- Smart Buildings TFDS: https://www.tensorflow.org/datasets/catalog/smart_buildings
