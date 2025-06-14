<!-- emailWorker.process('upcoming-invoice-email', async (job) => {
  const { to, subject, template, data } = job.data;

  const html =
    template === 'upcoming-invoice'
      ? upcomingInvoiceTemplate(data)
      : fallbackTemplate(data);

  await sendEmail({ to, subject, html });
}); -->
