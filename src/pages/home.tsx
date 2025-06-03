const Home = () => {
  return (
    <div className="min-h-screen w-full bg-[#FFFFFF] pt-7 px-16 max-sm:px-7">
      <div className="flex items-center justify-between py-3 px-4">
        <h1 className="text-3xl font-bold text-[#020717] ">Safeguardmedia</h1>
        <div className="flex justify-between items-center gap-4">
          <p className=" text-[#020717] font-[400] text-lg">Features</p>
          <p className=" text-[#020717] font-[400] text-lg">Pricing</p>
          <p className=" text-[#020717] font-[400] text-lg">FAQs</p>
          <button className="bg-[#0F2FA3] text-white text-lg px-4 py-2 rounded-[45px] font-semibold text-center">
            Get Started
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;
