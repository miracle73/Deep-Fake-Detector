<div className="px-4 sm:px-6 pb-4 sm:pb-6 min-w-0">
  <div className="bg-white rounded-xl border border-gray-200 p-4 sm:p-6 w-full overflow-hidden">
    <div className="mb-6">
      <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
        Your Recent Analyses
      </h3>
      <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
        Review the results of your recently uploaded media files. Results are
        stored for 30 days.
      </p>
    </div>
    {/* Loading State */}
    {isHistoryLoading && (
      <div className="text-center py-8 sm:py-12 w-full">
        <div className="flex justify-center items-center mb-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
        <p className="text-sm text-gray-600">
          Loading your analysis history...
        </p>
      </div>
    )}

    {/* Error State */}
    {historyError && (
      <div className="text-center py-8 sm:py-12 w-full">
        <div className="flex justify-center items-center mb-4">
          <AlertCircle className="w-8 h-8 text-red-500" />
        </div>
        <h4 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
          Error Loading History
        </h4>
        <p className="text-xs sm:text-sm text-red-600">
          Unable to load your analysis history. Please try refreshing the page.
        </p>
      </div>
    )}

    {!isHistoryLoading &&
      !historyError &&
      (historyData?.data && historyData.data.length > 0 ? (
        <div className="w-full">
          {/* Desktop Table Header */}
          <div className="hidden md:grid grid-cols-12 gap-4 pb-3 border-b border-gray-200 text-sm font-medium text-gray-500 min-w-0">
            <div className="col-span-4">File name/thumbnail</div>
            <div className="col-span-3">Upload date/time</div>
            <div className="col-span-2">Status</div>
            <div className="col-span-2">Confidence score</div>
            <div className="col-span-1"></div>
          </div>

          {/* Analysis History Items */}
          <div className="space-y-3 mt-4 w-full">
            {historyData.data.map((analysis) => {
              // Helper function to get file type icon
              const getFileTypeIcon = (fileName: string | undefined) => {
                if (!fileName)
                  return <ImageIcon className="w-6 h-6 text-gray-600" />;

                const extension = fileName.split(".").pop()?.toLowerCase();
                switch (extension) {
                  case "mp4":
                  case "avi":
                  case "mov":
                    return <Video className="w-6 h-6 text-gray-600" />;
                  case "mp3":
                  case "wav":
                  case "aac":
                    return <AudioLines className="w-6 h-6 text-gray-600" />;
                  case "jpg":
                  case "jpeg":
                  case "png":
                  case "webp":
                  default:
                    return <ImageIcon className="w-6 h-6 text-gray-600" />;
                }
              };

              // Helper function to format date
              const formatDate = (dateString: string): string => {
                const date = new Date(dateString);
                return date.toLocaleDateString("en-US", {
                  month: "short",
                  day: "2-digit",
                  year: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                  hour12: true,
                });
              };

              // Helper function to get status badge
              const getStatusBadge = (status: string | undefined): string => {
                const baseClasses =
                  "px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap";
                switch (status?.toLowerCase()) {
                  case "authentic":
                  case "real":
                    return `${baseClasses} bg-green-100 text-green-800`;
                  case "uncertain":
                  case "inconclusive":
                    return `${baseClasses} bg-yellow-100 text-yellow-800`;
                  case "deepfake":
                  case "fake":
                  case "synthetic":
                    return `${baseClasses} bg-red-100 text-red-800`;
                  default:
                    return `${baseClasses} bg-gray-100 text-gray-800`;
                }
              };

              return (
                <div key={analysis._id} className="w-full">
                  {/* Desktop View */}
                  <div className="hidden md:grid grid-cols-12 gap-4 items-center py-3 hover:bg-gray-50 rounded-lg min-w-0 transition-colors">
                    <div className="col-span-4 flex items-center space-x-3 min-w-0">
                      <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-gray-100 flex-shrink-0">
                        {analysis.thumbnailUrl ? (
                          <img
                            src={analysis.thumbnailUrl}
                            alt="File thumbnail"
                            className="w-full h-full object-cover rounded-lg"
                            onError={(e) => {
                              // Fallback to icon if thumbnail fails to load
                              const target = e.target as HTMLImageElement;
                              target.style.display = "none";
                              target.nextElementSibling?.classList.remove(
                                "hidden"
                              );
                            }}
                          />
                        ) : null}
                        <div className={analysis.thumbnailUrl ? "hidden" : ""}>
                          {getFileTypeIcon(analysis.fileName)}
                        </div>
                      </div>
                      <span
                        className="text-sm font-medium text-gray-900 truncate"
                        title={
                          analysis.fileName || `Media_${analysis._id.slice(-8)}`
                        }
                      >
                        {analysis.fileName || `Media_${analysis._id.slice(-8)}`}
                      </span>
                    </div>
                    <div className="col-span-3 min-w-0">
                      <span className="text-sm text-gray-600 break-words">
                        {formatDate(analysis.uploadDate)}
                      </span>
                    </div>
                    <div className="col-span-2 min-w-0">
                      <span className={getStatusBadge(analysis.status)}>
                        {analysis.status || "Unknown"}
                      </span>
                    </div>
                    <div className="col-span-2 min-w-0">
                      <span className="text-sm font-medium text-gray-900">
                        {analysis.confidenceScore
                          ? `${Math.round(analysis.confidenceScore)}%`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="col-span-1"></div>
                  </div>

                  {/* Mobile View */}
                  <div className="md:hidden bg-gray-50 rounded-lg p-4 hover:bg-gray-100 w-full transition-colors">
                    <div className="flex items-start space-x-3 min-w-0">
                      <div className="w-12 h-12 rounded-lg flex items-center justify-center bg-gray-100 flex-shrink-0">
                        {analysis.thumbnailUrl ? (
                          <img
                            src={analysis.thumbnailUrl}
                            alt="File thumbnail"
                            className="w-full h-full object-cover rounded-lg"
                            onError={(e) => {
                              // Fallback to icon if thumbnail fails to load
                              const target = e.target as HTMLImageElement;
                              target.style.display = "none";
                              target.nextElementSibling?.classList.remove(
                                "hidden"
                              );
                            }}
                          />
                        ) : null}
                        <div className={analysis.thumbnailUrl ? "hidden" : ""}>
                          {getFileTypeIcon(analysis.fileName)}
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex flex-col space-y-2">
                          <div className="min-w-0">
                            <p
                              className="text-sm font-medium text-gray-900 truncate"
                              title={
                                analysis.fileName ||
                                `Media_${analysis._id.slice(-8)}`
                              }
                            >
                              {analysis.fileName ||
                                `Media_${analysis._id.slice(-8)}`}
                            </p>
                            <p className="text-xs text-gray-600 mt-1 break-words">
                              {formatDate(analysis.uploadDate)}
                            </p>
                          </div>
                          <div className="flex items-center justify-between gap-2 flex-wrap">
                            <span className={getStatusBadge(analysis.status)}>
                              {analysis.status || "Unknown"}
                            </span>
                            <span className="text-sm font-medium text-gray-900 flex-shrink-0">
                              {analysis.confidenceScore
                                ? `${Math.round(analysis.confidenceScore)}%`
                                : "N/A"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pagination - Show pagination info using API response */}
          {historyData.pagination && historyData.pagination.totalPages > 1 && (
            <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-200">
              {/* Items per page info - Using API pagination data */}
              <div className="text-sm text-gray-600">
                Showing{" "}
                {(historyData.pagination.currentPage - 1) *
                  historyData.pagination.itemsPerPage +
                  1}{" "}
                to{" "}
                {Math.min(
                  historyData.pagination.currentPage *
                    historyData.pagination.itemsPerPage,
                  historyData.pagination.totalItems
                )}{" "}
                of {historyData.pagination.totalItems} analyses
              </div>

              {/* Pagination controls */}
              <div className="flex items-center space-x-1 sm:space-x-2">
                {/* Previous button */}
                <button
                  className="p-1.5 sm:p-2 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={() => {
                    if (
                      historyData.pagination.hasPreviousPage &&
                      historyData.pagination.previousPage
                    ) {
                      setCurrentPage(historyData.pagination.previousPage);
                    }
                  }}
                  disabled={!historyData.pagination.hasPreviousPage}
                >
                  <ChevronLeft className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
                </button>

                {/* Dynamic page numbers */}
                <div className="flex space-x-1 overflow-x-auto">
                  {getPageNumbers().map((page, index) =>
                    page === "ellipsis" ? (
                      <span
                        key={`ellipsis-${index}`}
                        className="px-2 py-1 text-gray-400"
                      >
                        ...
                      </span>
                    ) : (
                      <button
                        key={page}
                        className={`w-6 h-6 sm:w-8 sm:h-8 rounded text-xs sm:text-sm font-medium flex-shrink-0 transition-colors ${
                          currentPage === page
                            ? "bg-[#0F2FA3] text-white"
                            : "text-gray-600 hover:bg-gray-100"
                        }`}
                        onClick={() => {
                          setCurrentPage(page as number);
                          // No need to manually refetch - RTK Query will automatically
                          // refetch when the page parameter changes in useGetAnalysisHistoryQuery
                        }}
                      >
                        {page}
                      </button>
                    )
                  )}
                </div>

                {/* Next button */}
                <button
                  className="p-1.5 sm:p-2 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={() => {
                    if (
                      historyData.pagination.hasNextPage &&
                      historyData.pagination.nextPage
                    ) {
                      setCurrentPage(historyData.pagination.nextPage);
                    }
                  }}
                  disabled={!historyData.pagination.hasNextPage}
                >
                  <ChevronRight className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        // No Analyses Yet - Empty State
        <div className="text-center py-8 sm:py-12 w-full">
          <div className="flex justify-center items-center mb-4">
            <NoAnalysisYet />
          </div>
          <h4 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
            No Analyses Yet!
          </h4>
          <div className="space-y-1 max-w-md mx-auto">
            <p className="text-xs sm:text-sm text-gray-600">
              Upload your first video, audio, or image file to check for
              manipulation.
            </p>
            <p className="text-xs sm:text-sm text-gray-600">
              Our model will provide a quick and clear assessment.
            </p>
          </div>
        </div>
      ))}
  </div>
</div>;
