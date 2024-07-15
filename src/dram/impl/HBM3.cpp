#include "dram/dram.h"
#include "dram/lambdas.h"
// Considering QDR DQ pins, we double dq pins and halve burst length. So, rate 2 actually means 4 Gbps DQs for HBM3.

namespace Ramulator {

class HBM3 : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, HBM3, "HBM3", "HBM3 Device Model")

  public:
    inline static const std::map<std::string, Organization> org_presets = {
      // DQ for Pseudo Channel
      // 1/2/3/4R means 1/2/3/4 ranks for 4/8/12/16-Hi stack
      // We refer to JEDEC Standard (JESD238A).
      //   name          density    DQ    Ch Pch  Ra  Bg  Ba   Ro     Co
      {"HBM3_2Gb_1R",    {2<<10,    32,  {1,  2,  1,  4,  4, 1<<13,  1<<5}}},
      {"HBM3_4Gb_1R",    {4<<10,    32,  {1,  2,  1,  4,  4, 1<<14,  1<<5}}},
      {"HBM3_8Gb_1R",    {8<<10,    32,  {1,  2,  1,  4,  4, 1<<15,  1<<5}}},
      {"HBM3_4Gb_2R",    {4<<10,    32,  {1,  2,  2,  4,  4, 1<<13,  1<<5}}},
      {"HBM3_8Gb_2R",    {8<<10,    32,  {1,  2,  2,  4,  4, 1<<14,  1<<5}}},
      {"HBM3_16Gb_2R",   {16<<10,   32,  {1,  2,  2,  4,  4, 1<<15,  1<<5}}},
      {"HBM3_6Gb_3R",    {6<<10,    32,  {1,  2,  3,  4,  4, 1<<13,  1<<5}}},
      {"HBM3_12Gb_3R",   {12<<10,   32,  {1,  2,  3,  4,  4, 1<<14,  1<<5}}},
      {"HBM3_24Gb_3R",   {24<<10,   32,  {1,  2,  3,  4,  4, 1<<15,  1<<5}}},
      {"HBM3_8Gb_4R",    {8<<10,    32,  {1,  2,  4,  4,  4, 1<<13,  1<<5}}},
      {"HBM3_16Gb_4R",   {16<<10,   32,  {1,  2,  4,  4,  4, 1<<14,  1<<5}}},
      {"HBM3_32Gb_4R",   {32<<10,   32,  {1,  2,  4,  4,  4, 1<<15,  1<<5}}},
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      //   name       rate   nBL  nCL  nRCDRD  nRCDWR  nRP  nRAS  nRC  nWR  nRTPS  nRTPL  nCWL  nCCDS  nCCDL  nRRDS  nRRDL  nWTRS  nWTRL  nRTW  nFAW  nRFC  nRFCSB  nREFI  nREFISB  nRREFD  tCK_ps
      {"HBM3_4.8Gbps",  {4800,   2,  17,   17,     17,    17,   41,  58,  20,    5,     8,    5,    2,      4,     2,     4,     8,    10,    3,    36,   -1,   240,   4680,     -1,     10,   1200}},
      {"HBM3_5.2Gbps",  {5200,   2,  19,   19,     19,    19,   45,  63,  21,    6,     8,    6,    2,      4,     2,     4,     8,    11,    3,    39,   -1,   260,   5070,     -1,     11,   1300}},
      {"HBM3_5.6Gbps",  {5600,   2,  20,   20,     20,    20,   48,  68,  23,    6,     9,    6,    2,      4,     2,     4,     9,    12,    3,    42,   -1,   280,   5460,     -1,     12,   1400}},
      {"HBM3_6.0Gbps",  {6000,   2,  21,   21,     21,    21,   51,  72,  24,    6,     9,    6,    2,      4,     2,     4,     9,    12,    3,    45,   -1,   300,   5850,     -1,     12,   1500}},
      {"HBM3_6.4Gbps",  {6400,   2,  23,   23,     23,    23,   55,  77,  26,    7,    10,    7,    2,      4,     2,     4,    10,    13,    3,    48,   -1,   320,   6240,     -1,     13,   1600}},
      // TODO: Find more sources on HBM3 timings...
      // We could not find released HBM3 timing parameters. So, we mostly refer to the absolute value (ns) of the HBM2 timing parameters in DRAMSim3 (https://github.com/umd-memsys/DRAMsim3/blob/master/configs/HBM2_8Gb_x128.ini, commit: 29817593b3389f1337235d63cac515024ab8fd6e)
    };


  /************************************************
   *                Organization
   ***********************************************/   
    const int m_internal_prefetch_size = 8;

    inline static constexpr ImplDef m_levels = {
      "channel", "pseudochannel", "rank", "bankgroup", "bank", "row", "column",    
    };


  /************************************************
   *             Requests & Commands
   ***********************************************/
    inline static constexpr ImplDef m_commands = {
      "ACT", 
      "PRE", "PREA",
      "RD",  "WR",  "RDA",  "WRA",
      "REFab", "REFsb",
      "RFMab", "RFMsb"
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT",   "row"},
        {"PRE",   "bank"},    {"PREA",   "channel"},
        {"RD",    "column"},  {"WR",     "column"}, {"RDA",   "column"}, {"WRA",   "column"},
        {"REFab", "channel"}, {"REFsb",  "bank"},
        {"RFMab", "channel"}, {"RFMsb",  "bank"},
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
                // open?   close?   access?  refresh?
        {"ACT",   {true,   false,   false,   false}},
        {"PRE",   {false,  true,    false,   false}},
        {"PREA",  {false,  true,    false,   false}},
        {"RD",    {false,  false,   true,    false}},
        {"WR",    {false,  false,   true,    false}},
        {"RDA",   {false,  true,    true,    false}},
        {"WRA",   {false,  true,    true,    false}},
        {"REFab", {false,  false,   false,   true }},
        {"REFsb", {false,  false,   false,   true }},
        {"RFMab", {false,  false,   false,   true }},
        {"RFMsb", {false,  false,   false,   true }},
      }
    );

    inline static constexpr ImplDef m_requests = {
      "read", "write", "all-bank-refresh", "per-bank-refresh", "all-bank-rfm", "per-bank-rfm"
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        {"read", "RD"}, {"write", "WR"}, {"all-bank-refresh", "REFab"}, {"per-bank-refresh", "REFsb"}, 
        {"all-bank-rfm", "RFMab"}, {"per-bank-rfm", "RFMsb"}, 
      }
    );

   
  /************************************************
   *                   Timing
   ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL", "nCL", "nRCDRD", "nRCDWR", "nRP", "nRAS", "nRC", "nWR", "nRTPS", "nRTPL", "nCWL",
      "nCCDS", "nCCDL",
      "nRRDS", "nRRDL",
      "nWTRS", "nWTRL",
      "nRTW",
      "nFAW",
      "nRFC", "nRFCSB", "nREFI", "nREFISB", "nRREFD",
      "tCK_ps"
    };


  /************************************************
   *                 Node States
   ***********************************************/
    inline static constexpr ImplDef m_states = {
       "Opened", "Closed", "N/A"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",       "N/A"}, 
        {"pseudochannel", "N/A"}, 
        {"rank",         "N/A"}, // SID
        {"bankgroup",     "N/A"},
        {"bank",          "Closed"},
        {"row",           "Closed"},
        {"column",        "N/A"},
      }
    );

  public:
    struct Node : public DRAMNodeBase<HBM3> {
      Node(HBM3* dram, Node* parent, int level, int id) : DRAMNodeBase<HBM3>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;


  public:
    void tick() override {
      m_clk++;
    };

    void init() override {
      RAMULATOR_DECLARE_SPECS();
      set_organization();
      set_timing_vals();

      set_actions();
      set_preqs();
      set_rowhits();
      set_rowopens();
      
      create_nodes();
    };

    void issue_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      m_channels[channel_id]->update_timing(command, addr_vec, m_clk);
      m_channels[channel_id]->update_states(command, addr_vec, m_clk);
    };

    int get_preq_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->get_preq_command(command, addr_vec, m_clk);
    };

    bool check_ready(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_ready(command, addr_vec, m_clk);
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_rowbuffer_hit(command, addr_vec, m_clk);
    };

  private:
    void set_organization() {
      // Channel width
      m_channel_width = param_group("org").param<int>("channel_width").default_val(32);

      // Organization
      m_organization.count.resize(m_levels.size(), -1);

      // Load organization preset if provided
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
        if (org_presets.count(*preset_name) > 0) {
          m_organization = org_presets.at(*preset_name);
        } else {
          throw ConfigurationError("Unrecognized organization preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Override the preset with any provided settings
      if (auto dq = param_group("org").param<int>("dq").optional()) {
        m_organization.dq = *dq;
      }

      for (int i = 0; i < m_levels.size(); i++){
        auto level_name = m_levels(i);
        if (auto sz = param_group("org").param<int>(level_name).optional()) {
          m_organization.count[i] = *sz;
        }
      }

      if (auto density = param_group("org").param<int>("density").optional()) {
        m_organization.density = *density;
      }

      // Sanity check: is the calculated channel density the same as the provided one?
      size_t _density = size_t(m_organization.count[m_levels["pseudochannel"]]) *
                        size_t(m_organization.count[m_levels["rank"]]) *
                        size_t(m_organization.count[m_levels["bankgroup"]]) *
                        size_t(m_organization.count[m_levels["bank"]]) *
                        size_t(m_organization.count[m_levels["row"]]) *
                        size_t(m_organization.count[m_levels["column"]]) *
                        size_t(m_organization.dq) *
                        size_t(m_internal_prefetch_size);
      _density >>= 20;
      if (m_organization.density != _density) {
        throw ConfigurationError(
            "Calculated {} channel density {} Mb does not equal the provided density {} Mb!", 
            get_name(),
            _density, 
            m_organization.density
        );
      }

    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);

      // Load timing preset if provided
      bool preset_provided = false;
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        if (timing_presets.count(*preset_name) > 0) {
          m_timing_vals = timing_presets.at(*preset_name);
          preset_provided = true;
        } else {
          throw ConfigurationError("Unrecognized timing preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Check for rate (in MT/s), and if provided, calculate and set tCK (in picosecond)
      if (auto dq = param_group("timing").param<int>("rate").optional()) {
        if (preset_provided) {
          throw ConfigurationError("Cannot change the transfer rate of {} when using a speed preset !", get_name());
        }
        m_timing_vals("rate") = *dq;
      }
      int tCK_ps = 1E6 / (m_timing_vals("rate") / 4); // QDR DQ pins
      m_timing_vals("tCK_ps") = tCK_ps;

      // Refresh timings
      // tRFC table (unit is nanosecond!)
      constexpr int tRFC_TABLE[1][8] = {
      //  2Gb   4Gb   6Gb   8Gb   12Gb  16Gb  24Gb  32Gb
        { 160,  260,  310,  350,  410,  450,  610,  650},
      };

      // tRFC table (unit is nanosecond!)
      constexpr int tREFISB_TABLE[1][8] = {
      //  4-Hi   8-Hi   12-Hi  16-Hi
        { 244,   122,    82,    61},
      };

      int density_id = [](int density_Mb) -> int { 
        switch (density_Mb) {
          case 2048:  return 0;
          case 4096:  return 1;
          case 6144:  return 2;
          case 8192:  return 3;
          case 12288: return 4;
          case 16384: return 5;
          case 24576: return 6;
          case 32768: return 7;
          default:    return -1;
        }
      }(m_organization.density);

      m_timing_vals("nRFC")  = JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);
      m_timing_vals("nREFISB")  = JEDEC_rounding(tREFISB_TABLE[0][m_organization.count[m_levels["rank"]]], tCK_ps);

      // Overwrite timing parameters with any user-provided value
      // Rate and tCK should not be overwritten
      for (int i = 1; i < m_timings.size() - 1; i++) {
        auto timing_name = std::string(m_timings(i));

        if (auto provided_timing = param_group("timing").param<int>(timing_name).optional()) {
          // Check if the user specifies in the number of cycles (e.g., nRCD)
          m_timing_vals(i) = *provided_timing;
        } else if (auto provided_timing = param_group("timing").param<float>(timing_name.replace(0, 1, "t")).optional()) {
          // Check if the user specifies in nanoseconds (e.g., tRCD)
          m_timing_vals(i) = JEDEC_rounding(*provided_timing, tCK_ps);
        }
      }

      // Check if there is any uninitialized timings
      for (int i = 0; i < m_timing_vals.size(); i++) {
        if (m_timing_vals(i) == -1) {
          throw ConfigurationError("In \"{}\", timing {} is not specified!", get_name(), m_timings(i));
        }
      }      

      // Set read latency
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

      // Populate the timing constraints
      #define V(timing) (m_timing_vals(timing))
      populate_timingcons(this, {
          /*** Channel ***/ 
          /// 2-cycle ACT command (for row commands)
          {.level = "channel", .preceding = {"ACT"}, .following = {"ACT", "PRE", "PREA", "REFab", "REFsb", "RFMab", "RFMsb"}, .latency = 2},

          /*** Pseudo Channel ***/ 
          // CAS <-> CAS
          /// Data bus occupancy
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nBL")},
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nBL")},
          /// CAS <-> PREA
          {.level = "pseudochannel", .preceding = {"RD"}, .following = {"PREA"}, .latency = V("nRTPS")},
          {.level = "pseudochannel", .preceding = {"WR"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},          
          /// RAS <-> RAS
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"PREA"}, .latency = V("nRAS")},          
          {.level = "pseudochannel", .preceding = {"PREA"}, .following = {"ACT"}, .latency = V("nRP")},          
          /// RAS <-> REF
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"REFab", "RFMab"}, .latency = V("nRC")},          
          {.level = "pseudochannel", .preceding = {"PRE", "PREA"}, .following = {"REFab", "RFMab"}, .latency = V("nRP")},          
          {.level = "pseudochannel", .preceding = {"RDA"}, .following = {"REFab", "RFMab"}, .latency = V("nRP") + V("nRTPS")},          
          {.level = "pseudochannel", .preceding = {"WRA"}, .following = {"REFab", "RFMab"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},          
          {.level = "pseudochannel", .preceding = {"REFab", "RFMab"}, .following = {"ACT", "REFsb", "RFMsb"}, .latency = V("nRFC")},          


          /*** Rank (or different BankGroup) (Table 3 — Array Access Timings Counted Individually Per Pseudo Channel, JESD-235C) ***/  
          // RAS <-> RAS
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},
          /// 4-activation window restriction
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},

          /// ACT actually happens on the 2-nd cycle of ACT, so +1 cycle to nRRD
          {.level = "rank", .preceding = {"ACT"}, .following = {"REFsb", "RFMsb"}, .latency = V("nRRDS") + 1},
          /// nRREFD is the latency between REFsb <-> REFsb to *different* banks
          {.level = "rank", .preceding = {"REFsb", "RFMsb"}, .following = {"REFsb", "RFMsb"}, .latency = V("nRREFD")},
          /// nRREFD is the latency between REFsb <-> ACT to *different* banks. -1 as ACT happens on its 2nd cycle
          {.level = "rank", .preceding = {"REFsb", "RFMsb"}, .following = {"ACT"}, .latency = V("nRREFD") - 1},

          // CAS <-> CAS
          /// nCCDS is the minimal latency for column commands 
          {.level = "rank", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDS")},
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDS")},
          /// RD <-> WR, Minimum Read to Write, Assuming tWPRE = 1 tCK                          
          {.level = "rank", .preceding = {"RD", "RDA"}, .following = {"WR", "WRA"}, .latency = V("nCL") + V("nBL") + 2 - V("nCWL")},
          /// WR <-> RD, Minimum Read after Write
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRS")},
          /// RAS <-> RAS
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},          
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},          
          {.level = "rank", .preceding = {"ACT"}, .following = {"PREA"}, .latency = V("nRAS")},          
          {.level = "rank", .preceding = {"PREA"}, .following = {"ACT"}, .latency = V("nRP")},          

          /*** Same Bank Group ***/ 
          /// CAS <-> CAS
          {.level = "bankgroup", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRL")},
          /// RAS <-> RAS
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDL")},  
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"REFsb", "RFMsb"}, .latency = V("nRRDL") + 1},  
          {.level = "bankgroup", .preceding = {"REFsb", "RFMsb"}, .following = {"ACT"}, .latency = V("nRRDL") - 1},  

          {.level = "bank", .preceding = {"RD"},  .following = {"PRE"}, .latency = V("nRTPS")},  

          /*** Bank ***/ 
          {.level = "bank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"RD", "RDA"}, .latency = V("nRCDRD")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"WR", "WRA"}, .latency = V("nRCDWR")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"PRE"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"PRE"}, .following = {"ACT"}, .latency = V("nRP")},  
          {.level = "bank", .preceding = {"RD"},  .following = {"PRE"}, .latency = V("nRTPL")},  
          {.level = "bank", .preceding = {"WR"},  .following = {"PRE"}, .latency = V("nCWL") + V("nBL") + V("nWR")},  
          {.level = "bank", .preceding = {"RDA"}, .following = {"ACT", "REFsb", "RFMsb"}, .latency = V("nRTPL") + V("nRP")},  
          {.level = "bank", .preceding = {"WRA"}, .following = {"ACT", "REFsb", "RFMsb"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},  
        }
      );
      #undef V

    };

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));

      // Pseudo Channel Actions
      m_actions[m_levels["channel"]][m_commands["PREA"]] = Lambdas::Action::Channel::PREA<HBM3>;

      // Bank Actions
      m_actions[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Action::Bank::ACT<HBM3>;
      m_actions[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Action::Bank::PRE<HBM3>;
      m_actions[m_levels["bank"]][m_commands["RDA"]] = Lambdas::Action::Bank::PRE<HBM3>;
      m_actions[m_levels["bank"]][m_commands["WRA"]] = Lambdas::Action::Bank::PRE<HBM3>;
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));

      // Pseudo Channel Preqs
      m_preqs[m_levels["channel"]][m_commands["REFab"]] = Lambdas::Preq::Channel::RequireAllBanksClosed<HBM3>;

      // Bank Preqs
      m_preqs[m_levels["bank"]][m_commands["REFsb"]] = Lambdas::Preq::Bank::RequireBankClosed<HBM3>;
      m_preqs[m_levels["bank"]][m_commands["RD"]] = Lambdas::Preq::Bank::RequireRowOpen<HBM3>;
      m_preqs[m_levels["bank"]][m_commands["WR"]] = Lambdas::Preq::Bank::RequireRowOpen<HBM3>;
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowhits[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowHit::Bank::RDWR<HBM3>;
      m_rowhits[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowHit::Bank::RDWR<HBM3>;
    }


    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowopens[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowOpen::Bank::RDWR<HBM3>;
      m_rowopens[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowOpen::Bank::RDWR<HBM3>;
    }


    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        Node* channel = new Node(this, nullptr, 0, i);
        m_channels.push_back(channel);
      }
    };
};


}        // namespace Ramulator